The discussion on the vector dialect includes a description of how n-D vectors
are expected to be lowered to LLVM. In particular, it compares two views of how
to lower an n-D vector to LLVM, that being as nested aggregate, or stacked 1-D
vectors

!llvm."[s_0x[s_1x[...<s_{n-1}xf32>]]]">

or by flattening to a 1-D vector.

!llvm<"(s_0*...*s_{n-1})xfloat">

With pros/cons. First focusing on the aggregated view, because that is what is
adopted as the default at the end of the discussion, this makes a shape_cast/
reshape on a vector "free" if the affected dimensions are not inner-most. For
targets with n-D w/ n > 1 hardware registers, reshapes are only "free" if not
on the n inner-most dimensions. Because the builtin vector type is an abstract
SIMD type, reshapes always have some cost associated with them until lowered.

What is a virtual vector? A virtual SIMD type that maps from indices to elements
such that those elements have specific locations in a vector element space with
infinite width and dimensionality, or in other words, a virtual SIMD register
file. With this, if either the linearized bitvector of elements change as a
result of an operation, or the locations of those elements change within the
virtual SIMD register, then the operation has non-zero cost. With this, the
lowerings of operations in the vector dialect is the process of emulating that
virtual SIMD machine.

Emulating the virtual SIMD machine:

To target LLVM with the vector dialect, all vectors must be lowered to `1-D`.
This is currently managed with unrolling + a series of lowering patterns that
break down each given operation into a collection of 1-D vectors, which are
then lowered to LLVM. (oversimplifying, but that's my basic understanding of
the flow).

To target SPIR-V, there is an additional requirement that all vectors are <= 4
elements wide, requiring additional unrolling of the inner most dimension.

VirtualSIMDToNativeSIMD:
A VV -> VV conversion pass that maps operations on virtual vectors to a native
vector size. Similar to vector unrolling, but uses dialect conversion infra
to ensure that no composite vectors are left over. For SPIR-V, this would be a
native vector size of `4`, and for LLVM, we might still want to do this to avoid
relying on LLVM magic to do the unrolling for you.

This allows for partial conversions as well, if unrolling is desired on only a
portion of the IR for cases like WMMA, which might then get mapped to its own
target specific types before later lowerings.

The simplest implementation of the pass would just partition all vectors
gridwise according to the native vector size, for example:

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

VirtualSIMDToVirtualSIMT:
Another VV -> VV conversion pass that maps vectors from SIMD to SIMT. This views
threadblocks on GPUs as a single SIMD unit where each thread in a 3-D grid is a
location in a 3-D SIMD register where the shape of that register is specified
by the workgroup size of the kernel (e.g. [dim_x, dim_y, dim_z]). This means if
we take the same "stacked vector" view of the threadblock, going from SIMD to
SIMT (i.e. distribution) is just a matter of mapping the appropriate vector
dimensions to threads.

!vector_simt_t = vector< ... x "T_Z" x "T_Y" x "T_X" x  ? xi32>
!vector_simt_t = vector<  11 x   1   x   8   x  64   x 16 xi32>

For example, take the above. Assume we have one dimension for the SIMD local
to each thread (the inner most dimension) and then `3` dimensions for the grid
of threads. To move from SIMD to SIMT with a workgroup size of [64, 8, 1], we
simply drop the middle 3 dimensions

vector<11x1x8x64x16xi32> -> vector<11x16xi32>

Then, this becomes a stack of `11` vector<16xi32> thread-local vectors that
would get further unrolled.

Each of these passes can be applied multiple times in a full codegen flow. For
example, applying "VirtualSIMTToNativeSIMD" before "VirtualSIMDToVirtualSIMT" is
equivalent to overprovisioning. Similarly "VirtualSIMDToVirtualSIMT" could be
applied multiple times in a GPU codegen flow (e.g. warps and threads).

## Pre-emptive Discussion

The above models SIMD to SIMT distribution using only shape + rank information
on the vector type. This has a few potential pros/cons.

Pros:
 - Maps the idea of SIMT -> SIMD to the existing vector type, and with the
   appropriate op additions, should be able to represent at the SIMD level
   most programs we can expect to see (supplemented by target specific vector
   dialects).
 - Is closer to the way the program will actually be executed (i.e. a
   threadblock with a fixed grid size).
 - More reuse with non-SIMT targets
 - Puts more responsibility on the operations to be consistent with how they
   transform the layout.

Cons:
 - Managing the implicit requirements imposed by the rank of the SIMD (where
   the inner n dimensions map implicitly to the SIMD unit) require some shape
   gymnastics and puts a lot of burden on the layout analysis (left out of scope
   for this writeup).
 - It may prove to be difficult in some cases to reconcile the dimensionality of
   the problem with the fixed dimensionality of the vector type.
 - Contraction is hard.

One way to address the cons might be with a new layout based vector type, closer
to a tensor (although maybe the answer is just tensor at that point?). This can
come with a higher level vector dialect where additions are carefully considered
based on which existing vector ops make sense for layout vectors.

// -----
// List of proposed operations
As a continuation of the above, this is a list of proposed operations to
supplement the above passes.

- vector.expand
- vector.collapse
- vector.squeeze
- vector.unsqueeze
- vector.pack
- vector.unpack
- vector.packed_transfer_read
- vector.packed_transfer_write
- vector.replicate
- vector.group_reduce (?) allreduce (?)
- vector.generic

// -----

"vector.expand" reorganizes a vector in a higher dimensionality. The linearized
logical ordering of elements remains the same before and after the expand. The
scan product of the expanded shape must be a strict multi-superset of the scan
product of the original shape.

Examples:
vector.expand %0 : vector<4x16xf32> to vector<2x2x4x8xf32>
vector.expand %0 : vector<f32> to vector<1x1xf32>
vector.expand %0 : vector<8xf32> to vector<4x2xf32>

Illegal:
vector.expand %0 : vector<6x4xf32> to vector<2x6x2xf32>
vector.expand %0 : vector<6x4xf32> to vector<4x1x6xf32>

Rationale: vector.collapse (and transitively vector.shape_cast) is a distinct
enough operation that it warrants a distinct name (rather than a C++ function to
match on a vector.shape_cast, assuming the semantics of that op stay the same).
Taking the third 1-D case as an example, `vector.expand` that inserts an inner
dim of size 2 expands the 4 element vector into 4 2-element vectors, which after
lowering to all 1-D vectors would be 4 independent vectors. vector.collapse does
the opposite in taking the 4 independent vectors and shuffling them into one
vector.

// -----

"vector.collapse" reorganizes a vector in a lower dimensionality. The linearized
logical ordering of elements remains the same before and after the collapse. The
scan product of the original shape must be a strict multi-superset of the scan
product of the collapsed shape.

Examples:
vector.collapse %0 : vector<2x2x4x8xf32> to vector<2x8x8xf32>
vector.collapse %0 : vector<1x1xf32> to vector<f32>
vector.collapse %0 : vector<4x2xf32> to vector<8xf32>

Rationale: Inverse of vector.expand

// -----

"vector.squeeze" is a special case of vector.collapse that only drops unit
dimensions.

Examples:
vector.squeeze %0 : vector<1x1xf32> to vector<f32>
vector.squeeze %0 : vector<4x1xf32> to vector<4xf32>

// -----

"vector.unsqueeze" is a special case of vector.expand that only adds unit
dimensions.

Examples:
vector.squeeze %0 : vector<f32> to vector<1x1xf32>
vector.squeeze %0 : vector<4xf32> to vector<4x1xf32>

// -----

"vector.pack" packs the given vector with a set of pack sizes. All pack ops can
be expressed as a `vector.expand` + a `vector.transpose`, or as a sequence of
`vector.extract_slice` + `vector.insert_slice` ops. The pack takes a list of
outer sizes of the same rank as the input vector. Each size indicates the tiling
factor for the associated dim, with `0` indicating that it isn't tiled. The
outer tile is then transposed to the outer most dimensions of the packed vector,
keeping the same order between outer tiles.

Rationale: Because it is functionally similar to `tensor.pack` in terms of the
way data is reorganized, this op is closer to structured vector unrolling. This
is based on the observation that by viewing a vector as a stack of `k-D` vectors
where `k` is the dimensionality of the SIMD currently being targeted, a reshape
on an inner dim is actually quite expensive as it implies shuffles of large
amounts of data, when really all that is happening is unrolling along the inner
most dimension.

Examples:
vector.pack %0[0, 2] : vector<8x6xf32> to vector<2x8x3xf32>
vector.pack %0[4, 2] : vector<8x6xf32> to vector<4x2x2x3xf32>
vector.pack %0[4, 0] : vector<8x6xf32> to vector<4x2x6xf32> // (vector.expand)

// -----

"vector.unpack" unpacks the given vector along the specified dimensions. This
can always be expressed as a `vector.transpose` + `vector.collapse`, or as a
sequence of `vector.extract_slice` + `vector.insert_slice` ops. The unpack
takes a list of dimensions with length at most half of the input vector rank.
These dimensions specify where to transpose the outer dimensions to before
collapsing.

Rationale: This is the inverse of vector.pack and can be seen as a structured
form of vector shuffle.

Examples:
vector.unpack %0[2] : vector<2x8x3xf32> to vector<8x6xf32>
vector.unpack %0[1, 2] : vector<4x2x2x3xf32> to vector<8x6xf32>
vector.unpack %0[1] : vector<4x2x6xf32> to vector<8x6xf32> // (vector.collapse)

// Alternate representation using the pack sizes directly
vector.unpack %0[0, 2] : vector<2x8x3xf32> to vector<8x6xf32>
vector.unpack %0[4, 2] : vector<4x2x2x3xf32> to vector<8x6xf32>
vector.unpack %0[4, 0] : vector<4x2x6xf32> to vector<8x6xf32> // (vector.collapse)

// -----

"vector.packed_transfer_read" (Write description/rationale TODO)

Examples TODO.

// -----

"vector.packed_transfer_write" (Write description/rationale TODO)

Examples TODO.

// -----

"vector.replicate" Similar to vector.broadcast, but along any dimension. Follows
numpy-style broadcasting rules to keep it rank invariant.

Examples TODO.

// -----

"vector.group_reduce" Reduces a vector along the specified dimensions with
the specified reduction type and broadcasts the result to the original
vector shape. This is a reflection of the fact that when doing parallel
reductions, some strategies naturally end up with all threads holding the same
values at the end (the broadcast happens as a part of the operation). Might be
better suited for the GPU dialect.

Examples TODO.

// -----

"vector.generic" (Write description/rationale TODO)

Examples TODO.

// -----

## Examples

This section walks through a couple examples showing why representing the full
SIMD of a threadblock earlier on in a codegen flow helps solve problems that
have historically, in my experience, been difficult and brittle at times.

1) Can decide how vectors are to be distributed on tensors rather than after
bufferization.

If we want to do layout analysis + distribution, currently that must happen
on buffers (tensors are not distributable) unless there is a way to keep the
vectors with layouts in IR.

2) Promotion of the LHS and RHS of a matmul to workgroup memory.

A typical strategy for matrix multiplication codegen on GPUs involves looping
over the K dimension of the problem and accumulating partial tiles of the
output. To take better advantage of the data reuse between the LHS and RHS of
the contraction, the LHS and RHS for each output tile is copied to workgroup
memory.

The way I've seen this done typically relies on sending some signal to
bufferization in the form of a `linalg.copy` or `bufferization.alloc_tensor`
to get the desired allocation + copy, and then those copies are later
distributed, however the distribution of values across threads is different
before and after the copy. Knowing how to distribute those copies, or even
what is a "copy to shared memory" can be difficult to determine. Consider the
following simple example with a 2-d grid of threads of shape [4, 4] with a
per-thread SIMD size of `1`. Thus a `vector<4x4x1xf32>` has each thread
holding a single value.

```
#contraction_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1, 0)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}
scf.for %i = %c0 to %K step %c16 iter_args = (%out = vector<4x4x1xf32>) {
  // Each thread holds a single row of the lhs and rhs needed to compute its
  // dot product in the matmul_transpose_b.
  %lhs_vec = vector.transfer_read %lhs[...]
             { permutation_map = affine_map<(d0, d1) -> (0, d0, d1) } : vector<4x4x4xf32>
  %rhs_vec = vector.transfer_read %rhs[...]
             { permutation_map = affine_map<(d0, d1) -> (d0, 0, d1) } : vector<4x4x4xf32>

  // matmul_transpose_b
  %res = vector.contract #contraction_trait %lhs, %rhs, %out : vector<4x4x1xf32>
  scf.yield %res : vector<4x4x1xf32>
}
```

Then, we just have to introduce copies on the un-broadcasted vector that happen
using all threads

```
#contraction_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

scf.for %i = %c0 to %K step %c4 iter_args = (%out = vector<4x4x1xf32>) {
  // Copy the lhs to a workgroup local tensor
  %lhs_empty = tensor.empty() : tensor<4x4xf32>
  %lhs_read = vector.transfer_read %lhs[...] : vector<4x4x1xf32>
  %lhs_write = vector.transfer_write %lhs_read to %lhs_empty

  // Copy the rhs to a workgroup local tensor
  %rhs_empty = tensor.empty() : tensor<4x4xf32>
  %rhs_read = vector.transfer_read %rhs[...] : vector<4x4x1xf32>
  %rhs_write = vector.transfer_write %rhs_read to %rhs_empty

  // Each thread holds a single row of the lhs and rhs needed to compute its
  // dot product in the matmul_transpose_b.
  %lhs_vec = vector.transfer_read %lhs_write[...]
             { permutation_map = affine_map<(d0, d1) -> (0, d0, d1) } : vector<4x4x4xf32>
  %rhs_vec = vector.transfer_read %rhs_write[...]
             { permutation_map = affine_map<(d0, d1) -> (d0, 0, d1) } : vector<4x4x4xf32>

  // matmul_transpose_b
  %res = vector.contract #contraction_trait %lhs, %rhs, %out : vector<4x4x1xf32>
  scf.yield %res : vector<4x4x1xf32>
}
```

Edit: There is probably an even more natural way to get the copies here, and
that is to start with IR like the following.

```
#contraction_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,
  affine_map<(d0, d1, d2) -> (d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}
scf.for %i = %c0 to %K step %c4 iter_args = (%out = vector<4x4xf32>) {
  // Each thread holds a single row of the lhs and rhs needed to compute its
  // dot product in the matmul_transpose_b.
  %lhs_vec = vector.transfer_read %lhs[...] : vector<4x4xf32>
  %rhs_vec = vector.transfer_read %rhs[...] : vector<4x4xf32>

  // matmul_transpose_b
  %res = vector.contract #contraction_trait %lhs, %rhs, %out : vector<4x4xf32>
  scf.yield %res : vector<4x4xf32>
}
```

Which is what we normally start with, and the process of mapping the contraction
to the threadblock SIMT shape would introduce the appropriate broadcasts as well
as copies to resolve the preferred layout between the reads and the contraction.

3) Pipelining + multi-buffering on tensors.

Now that we've setup the structure of the inner loop of our GEMM on tensors, we
can pipeline to try to hide the latency of the reads + copies. Consider a simple
depth-2 pipeline that puts the copies to "shared memory" in stage 0 and the
reads from "shared memory" + compute in stage 1.

```
#contraction_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

%lhs_empty = tensor.empty() : tensor<4x4xf32>
%rhs_empty = tensor.empty() : tensor<4x4xf32>
scf.for %i = %c0 to %K step %c16 iter_args = (%out = vector<4x4x1xf32>) {
  // Copy the lhs to a workgroup local tensor
  %lhs_read = vector.transfer_read %lhs[...] : vector<4x4x1xf32> # stage 0
  %lhs_write = vector.transfer_write %lhs_read to %lhs_empty # stage 0

  // Copy the rhs to a workgroup local tensor
  %rhs_read = vector.transfer_read %rhs[...] : vector<4x4x1xf32> # stage 0
  %rhs_write = vector.transfer_write %rhs_read to %rhs_empty # stage 0

  // Each thread holds a single row of the lhs and rhs needed to compute its
  // dot product in the matmul_transpose_b.
  %lhs_vec = vector.transfer_read %lhs_write[...] # stage 1
             { permutation_map = affine_map<(d0, d1) -> (0, d0, d1) } : vector<4x4x4xf32>
  %rhs_vec = vector.transfer_read %rhs_write[...] # stage 1
             { permutation_map = affine_map<(d0, d1) -> (d0, 0, d1) } : vector<4x4x4xf32>

  // matmul_transpose_b
  %res = vector.contract #contraction_trait %lhs, %rhs, %out : vector<4x4x1xf32> # stage 1
  scf.yield %res : vector<4x4x1xf32>
}
```

This will approximately pipeline to (with a depth of 2)

```
#contraction_accesses = [
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

%lhs_empty = tensor.empty() : tensor<4x4xf32>
%rhs_empty = tensor.empty() : tensor<4x4xf32>
scf.for %i = %c0 to %K step %c16 iter_args = (%out = vector<4x4x1xf32>,
                                              %lhs_copy_s-2 = tensor<4x4xf32>,
                                              %rhs_copy_s-2 = tensor<4x4xf32>,
                                              %lhs_copy_s-1 = tensor<4x4xf32>,
                                              %rhs_copy_s-1 = tensor<4x4xf32>) {
  // Each thread holds a single row of the lhs and rhs needed to compute its
  // dot product in the matmul_transpose_b.
  %lhs_vec = vector.transfer_read %lhs_copy_s-2[...]
             { permutation_map = affine_map<(d0, d1) -> (0, d0, d1) } : vector<4x4x4xf32>
  %rhs_vec = vector.transfer_read %rhs_copy_s-2[...]
             { permutation_map = affine_map<(d0, d1) -> (d0, 0, d1) } : vector<4x4x4xf32>

  // matmul_transpose_b
  %res = vector.contract #contraction_trait %lhs, %rhs, %out : vector<4x4x1xf32>

  // Copy the lhs to a workgroup local tensor
  %lhs_read = vector.transfer_read %lhs[...] : vector<4x4x1xf32>
  %lhs_write = vector.transfer_write %lhs_read to %lhs_empty

  // Copy the rhs to a workgroup local tensor
  %rhs_read = vector.transfer_read %rhs[...] : vector<4x4x1xf32>
  %rhs_write = vector.transfer_write %rhs_read to %rhs_empty
  scf.yield %res, %lhs_copy_s-1, %rhs_copy_s-1, %lhs_write, %rhs_write
            : vector<4x4x1xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}
```

Whether or not this bufferizes is a separate question, but we get
multi-buffering out of necessity, rather than relying on an earlier pass to
produce the correct amount of multi-buffering for the amount of pipelining
being done.

// -----
// Other potential operations.
- vector.relayout

"vector.relayout" is a general operation for changing the layout of a vector.
This is based on the fact that the rank of a vector is closely tied to how it is
laid out in the virtual vector, and thus introducing extra dimensions via
reshapes and transposing such dimensions is very expensive if not represented
with a single op. This op can be thought of in general as `vector.shape_cast`
+ `vector.transpose` + `vector.shape_cast`, however by restricting the context
of the op to cases where the input and output vectors are expected to have a
valid layout, it is sufficient to restrict to expand -> transpose -> collapse.

Example:
vector.relayout %0 to vector<8x2x3xf32> by perm = [1, 0, 2] : vector<8x6xf32> -> vector<16x3xf32>

Decomposes to:
%1 = vector.expand %0 : vector<8x6xf32> to vector<8x2x3xf32>
%2 = vector.transpose %1 [1, 0, 2] : vector<8x2x3xf32> to vector<2x8x3xf32>
%3 = vector.collapse %2 : vector<2x8x3xf32> to vector<16x3xf32>

SIMDToSIMD: w/ native shape (4, 4)
// This does the following shuffle

// Input as 4 vectors:
// [ 0, 1, 2, 3] [ 4, 5, ., .]
// [ 6, 7, 8, 9] [10,11, ., .]
// [12,13,14,15] [16,17, ., .]
// [18,19,20,21] [22,23, ., .]
//
// [24,25,26,27] [28,29, ., .]
// [30,31,32,33] [34,35, ., .]
// [36,37,38,39] [40,41, ., .]
// [42,43,44,45] [46,47, ., .]

// Output as 4 vectors:
// [ 0, 1, 2, .]
// [ 6, 7, 8, .]
// [12,13,14, .]
// [18,19,20, .]
//
// [24,25,26, .]
// [30,31,32, .]
// [36,37,38, .]
// [42,43,44, .]
//
// [ 3, 4, 5, .]
// [ 9,10,11, .]
// [15,16,17, .]
// [21,22,23, .]
//
// [27,28,29, .]
// [33,34,35, .]
// [39,40,41, .]
// [45,46,47, .]

/// In general, this op is probably too difficult to work with.
