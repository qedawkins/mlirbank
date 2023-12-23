// All vectors sit in a virtual register of infinite dimension and width

!vector_2d_t = vector<2x2xi32>
// [[a, b, ...]
//  [c, d, ...]
//  [., ., ...]]

// This is in contrast with
//  - Tensor, a generic value semantic container, can't really say
//    anything about the storage. Different consumers interpret them
//    differently
//  - MemRef, a reference to a region in memory. Reshapes just change
//    where the lines are drawn, and all interaction with the underlying
//    data happens with explicit reads/writes
//
// As a result of vectors being closely tied to the storage, reshapes are not
// "no-op" or "free" operations. For some architectures, especially those with
// only 1-D HW SIMD units, might expect to eventually fold them away, but that
// is not the semantics of the operation.
//
// From the perspective of the vector dialect at least, downstream might be
// doing whatever they want.

vector.collapse_shape ... : vector<2x3xi32> to vector<6xi32>
// [[a, b, c, ...]
//  [d, e, f, ...]
//  [., ., ., ...]]
//
//   |
//   v
//
// [[a, b, c, d, e, f, ...]]

vector.expand_shape ... : vector<6xi32> to vector<3x2xi32>
// [[a, b, c, d, e, f, ...]]
//
//   |
//   v
//
// [[a, b, ...]
//  [c, d, ...]
//  [e, f, ...]
//  [., ., ...]]

// The layout of the data in register changes!
// This is why shape_cast is a *required* operation for the vector dialect.
// Shape casting for n-D hardware registers is very heavy if linearizing first,
// as it splits the data out into multiple registers and then reconstructs it.

// How does this work when going from a virtual or logical register to a
// concrete register width? Simply partition the vector grid-wise according
// to the native vector width.

!vector_2d_t = vector<4x7xi32> to vector<2x2xi32>
// [[ 0, 1, 2, 3, 4, 5, 6, ...]
//  [ 7, 8, 9,10,11,12,13, ...]
//  [14,15,16,17,18,19,20, ...]
//  [21,22,23,24,25,26,27, ...]
//  [ ., ., ., ., ., ., ., ...]]
//
//   |
//   v
//
// [ 0, 1] [ 2, 3] [ 4, 5] [ 6, _]
// [ 7, 8] [ 9,10] [11,12] [13, _]
//
// [14,15] [16,17] [18,19] [20, _]
// [21,22] [23,24] [25,26] [27, _]

// How does this relate to SIMT? We can think of the registers per thread as
// slots in a larger SIMD register. The shape of that register is fixed for a
// given kernel, but is configurable by the programmer/compiler. For example,
// a workgroup size of [64, 4, 1] corresponds to a native SIMD shape of
// vector<1x4x64x<per_thread_SIMD_width>>. There can be multiple levels of SIMT
// as well, with varying support for shuffles. If you really wanted to do
// something (ill-advised), you could write workgroup-level vector code.

vector<"WG_Z/T_Z" x "WG_Y/T_Y" x "WG_X/T_X" x "T_Z" x "T_Y" x "T_X" x thread_simd_width>

// The rules for a vector like this, are any movement that happen across
// workgroup dimensions have to happen with the appropriate global
// synchronization. This is hardly much different from threads. You can shuffle
// data across threads in a workgroup with shared memory, and you can sometimes
// shuffle data between threads in a subgroup with special subgroup operations.
//
// Essentially, this is representing the dimensionality of the entire problem
// with a single vector representing the state of the data in registers across
// the entire hardware.
//
// (don't think too hard about the workgroup thing, we're not gonna do that :)
//
// With this formulation, going from SIMD -> SIMT (distribution) is literally
// just a dialect conversion that drops the appropriate dimensions from the
// vector types.
//
// Say that my distributed dimensions are 1 and 2 of my vector types with a
// SIMD shape of [8, 64] (slower varying -> faster varying). This does the
// following type conversion.

vector<{11}x[8x64]x<16>xi32> -> vector<11x16xi32>

// Each thread now holds a vector of shape 11x16. Now, say that the native SIMD
// width of the thread is 4. This does unrolling such that

vector<11x16xi32> -> 11x4xvector<4xi32>

// Note that this higher dimensional representation makes coalescing load automatic.

vector.high_dimensionality_load memref<4096xi32> to vector<2048xi32> into_shape vector<8x64x4xi32>

// Distributes to

// 8 x
vector.load memref<4096xi32> to vector<4xi32>
// across 64 threads


vector.transpose <64x4> to <4x64>
                  s tx h      s tx h
vector.transpose <1x64x4> to <4x64x1>
vector.shape_cast <4x64x1> to <1x64x4>

vector.reduction <64xi32> to <i32>
vector.broadcast <i32> to <64xi32>

vector.subgroup_reduce <64xi32> to <64xi32>

Passes:
  - SIMDVectorToSmallerSIMDVector
    - Does the gridwise partitioning of vector
      - Type converter to allow users to control how types are translated
  - SIMDVectorToSIMTVector
    - "Folds" away distributed dimensions

Ops:
 - vector.squeeze
 - vector.unsqueeze
 - vector.collapse_shape
 - vector.expand_shape
 - vector.packed_transfer_read
 - vector.unpacked_transfer_write
 - vector.subgroup_reduce
 - vector.special_matmul (idk)


%0 = vector<6xf32>
%1 = vector.shape_cast %0 : vector<6xf32> to vector<2x3xf32>
%2 = vector.group_reduction %1 : vector<2x3xf32> to vector<2x3xf32>
%3 = arith.addi %2, %1 : vector<2x3xf32>, vector<2x3xf32>
%4 = vector.shape_cast %3 : vector<2x3xf32> to vector<3x2xf32>
%5 = vector.group_reduction %4 : vector<3x2xf32> to vector<3x2xf32>
%6 = arith.addi %5, %4 : vector<3x2xf32>, vector<3x2xf32>
%7 = vector.shape_cast %6 : vector<3x2xf32> to vector<6xf32>
