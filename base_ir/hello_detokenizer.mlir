module @main {
  util.global private mutable @tokens_map : !util.list<!util.buffer>
  util.global private mutable @tokens_list : !util.list<!util.buffer>
  util.global private mutable @num_tokens : index

  util.initializer {
    %capacity = arith.constant 25 : index
    %0 = util.list.create %capacity : !util.list<!util.buffer>
    util.global.store %0, @tokens_list : !util.list<!util.buffer>
    %c0 = arith.constant 0 : index
    util.global.store %c0, @num_tokens : index
    util.initializer.return
  }

  util.initializer {
    %num_tokens = arith.constant 3 : index
    %0 = util.list.create %num_tokens : !util.list<!util.buffer>
    %c0 = arith.constant 0 : index
    %hello = util.buffer.constant : !util.buffer = "hello"
    util.list.set %0[%c0], %hello : !util.list<!util.buffer>

    %c1 = arith.constant 1 : index
    %world = util.buffer.constant : !util.buffer = "world"
    util.list.set %0[%c1], %world : !util.list<!util.buffer>

    %c2 = arith.constant 2 : index
    %exc = util.buffer.constant : !util.buffer = "!"
    util.list.set %0[%c2], %exc : !util.list<!util.buffer>
    util.global.store %0, @tokens_map : !util.list<!util.buffer>
    util.initializer.return
  }

  func.func public @reset() {
    %capacity0 = arith.constant 25 : index
    %0 = util.list.create %capacity0 : !util.list<!util.buffer>
    util.global.store %0, @tokens_list : !util.list<!util.buffer>
    return
  }

  func.func public @detokenize(%ids: tensor<?xi32>) -> !util.list<!util.buffer> {
    %lst = util.global.load @tokens_list : !util.list<!util.buffer>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %steps = tensor.dim %ids, %c0 : tensor<?xi32>

    %start = util.global.load @num_tokens : index
    %map = util.global.load @tokens_map : !util.list<!util.buffer>
    scf.for %arg0 = %start to %steps step %c1 {
      %id = tensor.extract %ids[%arg0] : tensor<?xi32>
      %id_idx = arith.index_cast %id : i32 to index
      %token = util.list.get %map[%id_idx] : !util.list<!util.buffer>
      util.list.set %lst[%arg0], %token : !util.list<!util.buffer>
      scf.yield
    }

    %new_num_tokens = arith.addi %start, %steps : index
    util.global.store %new_num_tokens, @num_tokens : index
    return %lst : !util.list<!util.buffer>
  }
}
