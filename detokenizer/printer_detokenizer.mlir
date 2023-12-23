module @main {
  util.global private @tokens_table = dense<[0, 5, 5, 5, 10, 1]> : tensor<6xi32>
  util.global private @tokens_map = "worldhello!" : !util.buffer
  util.global private mutable @buffer_len = 0 : index

  func.func private @append_and_grow(%arg0: !util.buffer, %word: !util.buffer) -> !util.buffer {
    %c0_0 = arith.constant 0 : index
    %c2_0 = arith.constant 2 : index
    %capacity = util.buffer.size %arg0 : !util.buffer
    %word_size = util.buffer.size %word : !util.buffer
    %size = util.global.load @buffer_len : index
    %new_size = arith.addi %word_size, %size : index
    %cmp = arith.cmpi sgt, %new_size, %capacity : index
    %new_buffer = scf.if %cmp -> (!util.buffer) {
      %s2 = arith.muli %new_size, %c2_0 : index
      %new = util.buffer.alloc uninitialized : !util.buffer{%s2}
      %zcmp = arith.cmpi ne, %size, %c0_0 : index
      scf.if %zcmp {
        util.buffer.copy %arg0[%c0_0], %new[%c0_0], %size : !util.buffer{%size} -> !util.buffer{%s2}
      }
      scf.yield %new : !util.buffer
    } else {
      scf.yield %arg0 : !util.buffer
    }

    %new_cap = util.buffer.size %new_buffer : !util.buffer
    util.buffer.copy %word[%c0_0], %new_buffer[%size], %word_size : !util.buffer{%word_size} -> !util.buffer{%new_cap}
    util.global.store %new_size, @buffer_len : index
    return %new_buffer : !util.buffer
  }

  func.func public @detokenize(%ids: tensor<?xi32>) {
    %tokens_table = util.global.load @tokens_table : tensor<6xi32>
    // %tokens_map = util.global.load @tokens_map : !util.buffer
    %tokens_map = util.buffer.constant : !util.buffer = "worldhello!"

    //%table_size = util.buffer.size %tokens_table : !util.buffer
    %map_size = util.buffer.size %tokens_map : !util.buffer

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %steps = tensor.dim %ids, %c0 : tensor<?xi32>

    %stdout = io_stream.console.stdout : !io_stream.handle

    %init = util.buffer.alloc uninitialized : !util.buffer{%c2}
    %text = scf.for %arg0 = %c0 to %steps step %c1 iter_args(%arg1 = %init) -> !util.buffer {
      %id = tensor.extract %ids[%arg0] : tensor<?xi32>
      %id_idx = arith.index_cast %id : i32 to index
      %id2 = arith.muli %id_idx, %c2 : index
      %id2p1 = arith.addi %id2, %c1 : index
      // %offset = util.buffer.load %tokens_table[%id2 for %c4] : !util.buffer{%table_size} -> i32
      // %size = util.buffer.load %tokens_table[%id2p1 for %c4] : !util.buffer{%table_size} -> i32
      %offset = tensor.extract %tokens_table[%id2] : tensor<6xi32>
      %offset_idx = arith.index_cast %offset : i32 to index
      %size = tensor.extract %tokens_table[%id2p1] : tensor<6xi32>
      %size_idx = arith.index_cast %size : i32 to index
      %word = util.buffer.slice %tokens_map[%offset_idx] : !util.buffer{%map_size} -> !util.buffer{%size_idx}

      io_stream.write.bytes(%stdout, %word, %c0, %size_idx) : (!io_stream.handle, !util.buffer, index, index) -> ()

      %appended = func.call @append_and_grow(%arg1, %word) : (!util.buffer, !util.buffer) -> !util.buffer

      %curr_size = util.global.load @buffer_len : index
      %curr_cap = util.buffer.size %appended : !util.buffer
      %app_slice = util.buffer.slice %appended[%c0] : !util.buffer{%curr_cap} -> !util.buffer{%curr_size}
      io_stream.write.bytes(%stdout, %app_slice, %c0, %curr_size) : (!io_stream.handle, !util.buffer, index, index) -> ()

      scf.yield %appended : !util.buffer
    }

    %text_capacity = util.buffer.size %text : !util.buffer
    %text_size = util.global.load @buffer_len : index
    %slice = util.buffer.slice %text[%c0] : !util.buffer{%text_capacity} -> !util.buffer{%text_size}

    io_stream.write.bytes(%stdout, %slice, %c0, %text_size) : (!io_stream.handle, !util.buffer, index, index) -> ()
    %newline = arith.constant 10 : i8  // \n
    io_stream.write.byte(%stdout, %newline) : (!io_stream.handle, i8) -> ()
    return
  }
}
