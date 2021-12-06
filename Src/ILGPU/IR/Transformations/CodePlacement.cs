// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                           Copyright (c) 2021 ILGPU Project
//                                    www.ilgpu.net
//
// File: CodePlacement.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.IR.Analyses;
using ILGPU.IR.Analyses.TraversalOrders;
using ILGPU.IR.Values;
using ILGPU.Util;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Dominators = ILGPU.IR.Analyses.Dominators<
    ILGPU.IR.Analyses.ControlFlowDirection.Forwards>;

namespace ILGPU.IR.Transformations
{
    /// <summary>
    /// Represents a global code placement phase that moves values a close as possible
    /// to their uses. This minimizes liveranges of all values in the program.
    /// </summary>
    /// <remarks>
    /// This placement transformation should be used in combination with the
    /// <see cref="LoopInvariantCodeMotion"/> transformation to use values out of loops.
    /// </remarks>
    public class CodePlacement : SequentialUnorderedTransformation
    {
        #region Nested Types

        /// <summary>
        /// A single entry during the placement process.
        /// </summary>
        private readonly struct PlacementEntry
        {
            public PlacementEntry(Value value, BasicBlock block)
            {
                Value = value;
                Block = block;
            }

            /// <summary>
            /// The value to be placed.
            /// </summary>
            public Value Value { get; }

            /// <summary>
            /// The intended initial basic block.
            /// </summary>
            public BasicBlock Block { get; }

            /// <summary>
            /// Returns the string representation of this entry for debugging purposes.
            /// </summary>
            public readonly override string ToString() => $"{Value} @ {Block}";
        }

        /// <summary>
        /// An abstract placer mode that setups insert position for given blocks.
        /// </summary>
        private interface IPlacerMode
        {
            /// <summary>
            /// Setups the value insert position for the given block builder.
            /// </summary>
            /// <param name="builder">The current block builder.</param>
            void SetupInsertPosition(BasicBlock.Builder builder);
        }

        /// <summary>
        /// Appends values by inserting them behind a all phi values.
        /// </summary>
        private readonly struct AppendMode : IPlacerMode
        {
            public AppendMode(in BasicBlockMap<(Value[], int)> blocks)
            {
                Blocks = blocks;
            }

            /// <summary>
            /// Returns the current basic block map.
            /// </summary>
            public BasicBlockMap<(Value[] Values, int NumPhis)> Blocks { get; }

            /// <summary>
            /// Setups the insert position according to the number of detected phi
            /// values in each block.
            /// </summary>
            /// <param name="builder">The current builder.</param>
            public readonly void SetupInsertPosition(BasicBlock.Builder builder) =>
                builder.InsertPosition = Blocks[builder.BasicBlock].NumPhis;
        }

        /// <summary>
        /// Inserts all values at the beginning of each block.
        /// </summary>
        private readonly struct InsertMode : IPlacerMode
        {
            /// <summary>
            /// Setups the insert position to point to the start of the block.
            /// </summary>
            /// <param name="builder">The current builder.</param>
            public readonly void SetupInsertPosition(BasicBlock.Builder builder) =>
                builder.SetupInsertPositionToStart();
        }

        /// <summary>
        /// An internal placement helper structure that manages all values to be placed.
        /// </summary>
        private ref struct Placer
        {
            /// <summary>
            /// The queue of all remaining entries to be placed.
            /// </summary>
            private readonly Queue<PlacementEntry> toPlace;

            /// <summary>
            /// The set of all values that have been placed.
            /// </summary>
            private readonly HashSet<Value> placed;

            /// <summary>
            /// Constructs a new placer instance.
            /// </summary>
            /// <param name="builder">The parent method builder.</param>
            /// <param name="dominators">The parent dominators.</param>
            /// <param name="capacity">The initial placement queue capacity.</param>
            public Placer(Method.Builder builder, Dominators dominators, int capacity)
            {
                Builder = builder;
                Dominators = dominators;
                toPlace = new Queue<PlacementEntry>(capacity);
                placed = new HashSet<Value>();
            }

            /// <summary>
            /// Returns the parent method builder.
            /// </summary>
            public Method.Builder Builder { get; }

            /// <summary>
            /// Returns the parent dominators.
            /// </summary>
            public Dominators Dominators { get; }

            /// <summary>
            /// Returns true if the given value has been placed.
            /// </summary>
            public readonly bool IsPlaced(Value value) => placed.Contains(value);

            /// <summary>
            /// Places this value and all of its dependencies recursively.
            /// </summary>
            /// <param name="value">The value to place.</param>
            /// <param name="mode">The current placing mode.</param>
            public readonly void PlaceRecursive<TMode>(Value value, TMode mode)
                where TMode : struct, IPlacerMode
            {
                // Check whether we have to skip the current value
                var current = new PlacementEntry(value, value.BasicBlock);
                if (!IsPlaced(value))
                {
                    // Place this value
                    value.Assert(!CanMoveValue(Builder.Method, current.Value));
                    PlaceDirect(current.Value, current.Block, mode);
                }

                // Place all children
                EnqueueChildren(current.Value, current.Block);
                while (toPlace.Count > 0)
                {
                    // Get the next value to be placed
                    current = toPlace.Dequeue();

                    // Check whether we have to skip this value
                    if (placed.Contains(current.Value) ||
                        !TryPlace(current, out var placementBlock))
                    {
                        // This value cannot be placed since all of its uses are not
                        // placed yet or this value needs to be skipped anyway.
                        continue;
                    }

                    // Place this value
                    PlaceDirect(current.Value, placementBlock, mode);
                }
            }

            /// <summary>
            /// Enqueues all child values of the given placement entry.
            /// </summary>
            /// <param name="value">The value to enqueue all children for.</param>
            /// <param name="placementBlock">
            /// The block into which the the value has been placed
            /// </param>
            private readonly void EnqueueChildren(Value value, BasicBlock placementBlock)
            {
                // Push all child values
                for (int i = value.Count - 1; i >= 0; --i)
                {
                    Value node = value[i];
                    // Skip values that cannot be moved here
                    if (!CanMoveValue(Builder.Method, node))
                        continue;

                    // Add the node for processing but use the current block
                    toPlace.Enqueue(new PlacementEntry(node, placementBlock));
                }
            }

            /// <summary>
            /// Tries to place the given entry while determining a proper placement block.
            /// </summary>
            /// <param name="entry">The placement entry.</param>
            /// <param name="placementBlock">
            /// The determined placement block (if any).
            /// </param>
            /// <returns>
            /// True, if the value could be placed given its operand conditions.
            /// </returns>
            private readonly bool TryPlace(
                in PlacementEntry entry,
                out BasicBlock placementBlock)
            {
                // Test whether can actually place the current value
                placementBlock = null;
                if (!CanPlace(entry.Value) || !CanMoveValue(Builder.Method, entry.Value))
                    return false;

                // Determine the actual placement block
                placementBlock = Dominators.GetImmediateCommonDominatorOfUses(
                    entry.Block,
                    entry.Value.Uses);

                // Push all child values
                EnqueueChildren(entry.Value, placementBlock);

                return true;
            }

            /// <summary>
            /// Places a value directly without placing its operands.
            /// </summary>
            /// <param name="value">The value to be placed.</param>
            /// <param name="placementBlock">
            /// The block into which the value will be placed.
            /// </param>
            /// <param name="mode">The current placing mode.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public readonly void PlaceDirect<TMode>(
                Value value,
                BasicBlock placementBlock,
                TMode mode)
                where TMode : struct, IPlacerMode
            {
                // Mark the current value as placed
                bool hasNotBeenPlaced = placed.Add(value);
                value.Assert(hasNotBeenPlaced);

                // Skip terminator values
                if (value is TerminatorValue)
                    return;

                // Move the value to the determined placement block
                value.BasicBlock = placementBlock;
                var blockBuilder = Builder[placementBlock];
                mode.SetupInsertPosition(blockBuilder);
                blockBuilder.Add(value);
            }

            /// <summary>
            /// Returns true if the current value can be placed now by checking all of
            /// its uses.
            /// </summary>
            /// <param name="value">The value to be placed.</param>
            /// <returns>True, if the value could be placed.</returns>
            private readonly bool CanPlace(Value value)
            {
                // Check of all of its uses
                foreach (Value use in value.Uses)
                {
                    if (!placed.Contains(use) && !(use is PhiValue))
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// Gathers phi values in all blocks and clears all block-internal lists.
        /// </summary>
        private readonly struct GatherValuesInBlock :
            IBasicBlockMapValueProvider<(Value[] Values, int NumPhis)>
        {
            public GatherValuesInBlock(Method.Builder builder, List<PhiValue> phiValues)
            {
                Builder = builder;
                PhiValues = phiValues;
            }

            /// <summary>
            /// Returns the parent method builder.
            /// </summary>
            public Method.Builder Builder { get; }

            /// <summary>
            /// Returns the list of all phi values.
            /// </summary>
            public List<PhiValue> PhiValues { get; }

            /// <summary>
            /// Determines an array of all values of the given block in post order.
            /// </summary>
            /// <param name="block">The current block.</param>
            /// <param name="traversalIndex">The current traversal index.</param>
            public readonly (Value[], int) GetValue(
                BasicBlock block,
                int traversalIndex)
            {
                // Track the number of phi values in this block
                int numPhis = 0;

                // Build an array of values to process
                var values = new Value[block.Count];

                // "Append" all values in reversed order
                for (int i = 0, e = block.Count; i < e; ++i)
                {
                    Value value = block[e - 1 - i];
                    if (value is PhiValue phiValue)
                    {
                        PhiValues.Add(phiValue);
                        ++numPhis;
                    }
                    values[i] = value;
                }

                // Clear the lists of this block
                Builder[block].ClearLists();

                return (values, numPhis);
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Returns true if the given value can be moved to a different block. This is
        /// the case when we reach a node without side effects or phi values.
        /// </summary>
        /// <param name="method">The parent method.</param>
        /// <param name="value">The value to test.</param>
        /// <returns>
        /// True, if the given value can be moved to a different block.
        /// </returns>
        private static bool CanMoveValue(Method method, Value value) =>
            value switch
            {
                Parameter _ => false,
                MemoryValue _ => false,
                MethodCall _ => false,
                PhiValue _ => false,
                TerminatorValue _ => false,
                _ => method == value.Method &&
                    !value.Uses.Any(new UseCollection.HasPhiUsesPredicate())
            };

        /// <summary>
        /// Applies an accelerator-specialization transformation.
        /// </summary>
        protected override bool PerformTransformation(
            IRContext context,
            Method.Builder builder)
        {
            // Initialize a new placer instance
            var dominators = builder.SourceBlocks.CreateDominators();
            var placer = new Placer(
                builder,
                dominators,
                dominators.CFG.Count << 2);

            // Iterate over all values and determine their actual position in post order
            var blocks = builder.SourceBlocks.AsOrder<PostOrder>();

            // Gather all values in the whole function to be placed
            var phiValues = new List<PhiValue>(blocks.Count);
            var blockMapping = blocks.CreateMap(
                new GatherValuesInBlock(
                    builder,
                    phiValues));

            // Do not move phi values to different blocks
            foreach (var phiValue in phiValues)
                placer.PlaceDirect(phiValue, phiValue.BasicBlock, new InsertMode());

            // Place all values that require explicit placement operations
            var appendMode = new AppendMode(blockMapping);
            foreach (var block in blocks)
            {
                var blockEntry = blockMapping[block];

                // Place the terminator
                placer.PlaceRecursive(block.Terminator, appendMode);

                // Place all values that have to be placed recursively
                foreach (var value in blockEntry.Values)
                {
                    if (CanMoveValue(builder.Method, value))
                        continue;

                    // Force a placement of these values as they will have either
                    // side effects or should be placed here to minimize live spans of
                    // values.
                    placer.PlaceRecursive(value, appendMode);
                }
            }

#if DEBUG
            // Once we have placed all live values, all remaining values which have not
            // been placed yet are dead. However, these values can lead to invalid
            // results of this transformation.
            foreach (var block in blocks)
            {
                var (values, _) = blockMapping[block];
                foreach (var value in values)
                    value.Assert(placer.IsPlaced(value));
            }
#endif

            return true;
        }

        #endregion
    }
}

