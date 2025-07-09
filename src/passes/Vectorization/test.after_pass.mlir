module {
  moore.module @md1(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    moore.assign %out, %in : l4
    %0 = moore.read %out : <l4>
    moore.output %0 : !moore.l4
  }
  moore.module @md2(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract %in from 0 : l4 -> l1
    %1 = moore.extract %in from 1 : l4 -> l1
    %2 = moore.extract %in from 2 : l4 -> l1
    %3 = moore.extract %in from 3 : l4 -> l1
    %4 = moore.concat %0, %1, %2, %3 : (!moore.l1, !moore.l1, !moore.l1, !moore.l1) -> l4
    moore.assign %out, %4 : l4
    %5 = moore.read %out : <l4>
    moore.output %5 : !moore.l4
  }
  moore.module @md3(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract %in from 1 : l4 -> l1
    %1 = moore.extract %in from 2 : l4 -> l1
    %2 = moore.extract %in from 3 : l4 -> l1
    %3 = moore.extract %in from 0 : l4 -> l1
    %4 = moore.concat %3, %2, %1, %0 : (!moore.l1, !moore.l1, !moore.l1, !moore.l1) -> l4
    moore.assign %out, %4 : l4
    %5 = moore.read %out : <l4>
    moore.output %5 : !moore.l4
  }
  moore.module @md4(out out : !moore.l8, in %in : !moore.l8) {
    %out = moore.net wire : <l8>
    %0 = moore.extract %in from 1 : l8 -> l1
    %1 = moore.extract %in from 2 : l8 -> l1
    %2 = moore.extract %in from 3 : l8 -> l1
    %3 = moore.extract %in from 0 : l8 -> l1
    %4 = moore.extract %in from 5 : l8 -> l1
    %5 = moore.extract %in from 4 : l8 -> l1
    %6 = moore.extract %in from 6 : l8 -> l1
    %7 = moore.extract %in from 7 : l8 -> l1
    %8 = moore.concat %7, %6, %5, %4, %3, %2, %1, %0 : (!moore.l1, !moore.l1, !moore.l1, !moore.l1, !moore.l1, !moore.l1, !moore.l1, !moore.l1) -> l8
    moore.assign %out, %8 : l8
    %9 = moore.read %out : <l8>
    moore.output %9 : !moore.l8
  }
}

