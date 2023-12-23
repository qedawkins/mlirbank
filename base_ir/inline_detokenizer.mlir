module @main {
  func.func public @detokenize(%ids: tensor<?xi32>) -> !util.list<!util.buffer> {
    %num_tokens = arith.constant 3 : index
    %map = util.list.create %num_tokens : !util.list<!util.buffer>
    %c0 = arith.constant 0 : index
    %hello = util.buffer.constant : !util.buffer = "hello"
    util.list.set %map[%c0], %hello : !util.list<!util.buffer>

    %c1 = arith.constant 1 : index
    %world = util.buffer.constant : !util.buffer = "world"
    util.list.set %map[%c1], %world : !util.list<!util.buffer>

    %c2 = arith.constant 2 : index
    %exc = util.buffer.constant : !util.buffer = "!"
    util.list.set %map[%c2], %exc : !util.list<!util.buffer>

    %capacity = arith.constant 25 : index
    %lst = util.list.create %capacity : !util.list<!util.buffer>

    %steps = tensor.dim %ids, %c0 : tensor<?xi32>

    scf.for %arg0 = %c0 to %steps step %c1 {
      %id = tensor.extract %ids[%arg0] : tensor<?xi32>
      %id_idx = arith.index_cast %id : i32 to index
      %token = util.list.get %map[%id_idx] : !util.list<!util.buffer>
      util.list.set %lst[%arg0], %token : !util.list<!util.buffer>
      scf.yield
    }

    return %lst : !util.list<!util.buffer>
  }
}
