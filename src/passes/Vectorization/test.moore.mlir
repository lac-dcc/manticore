module {
  moore.module @linear(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 3 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.read %out : <l4>
    moore.output %2 : !moore.l4
  }
}
