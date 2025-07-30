module {
  moore.module @pattern_recognition(out result : !moore.l4, in %a : !moore.l4, in %b : !moore.l4, in %sel : !moore.l1) {
    %result = moore.net wire : <l4>
    %0 = moore.not %sel : l1
    %1 = moore.conditional %sel : l1 -> l4 {
      moore.yield %a : l4
    } {
      moore.yield %b : l4
    }
    moore.assign %result, %1 : l4
    %2 = moore.read %result : <l4>
    moore.output %2 : !moore.l4
  }
}

