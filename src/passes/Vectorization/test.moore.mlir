module {
  moore.module @md1(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 3 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 2 : l4 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 1 : l4 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 0 : l4 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
  moore.module @md2(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 0 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 1 : l4 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 2 : l4 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 3 : l4 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
  moore.module @md3(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 0 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 3 : l4 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 2 : l4 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 1 : l4 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
  moore.module @md4(out out : !moore.l8, in %in : !moore.l8) {
    %out = moore.net wire : <l8>
    %0 = moore.extract_ref %out from 7 : <l8> -> <l1>
    %1 = moore.extract %in from 7 : l8 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 6 : <l8> -> <l1>
    %3 = moore.extract %in from 6 : l8 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 5 : <l8> -> <l1>
    %5 = moore.extract %in from 4 : l8 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 4 : <l8> -> <l1>
    %7 = moore.extract %in from 5 : l8 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.extract_ref %out from 3 : <l8> -> <l1>
    %9 = moore.extract %in from 0 : l8 -> l1
    moore.assign %8, %9 : l1
    %10 = moore.extract_ref %out from 2 : <l8> -> <l1>
    %11 = moore.extract %in from 3 : l8 -> l1
    moore.assign %10, %11 : l1
    %12 = moore.extract_ref %out from 1 : <l8> -> <l1>
    %13 = moore.extract %in from 2 : l8 -> l1
    moore.assign %12, %13 : l1
    %14 = moore.extract_ref %out from 0 : <l8> -> <l1>
    %15 = moore.extract %in from 1 : l8 -> l1
    moore.assign %14, %15 : l1
    %16 = moore.read %out : <l8>
    moore.output %16 : !moore.l8
  }
}
