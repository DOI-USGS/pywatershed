To compile the PRMS 5.2.1 binary to be used as reference for pywatershed in
its autotests, run the following commands in this directory

```shell
make clean
make FC=ifort CC=icc DBL_PREC=true
```

You may replace `ifort` and `icc` with the name of the fortran compiler in your
`$PATH`. Note that `DBL_PREC=true` is required for pywatershed tests.

Compiling replaces `bin/prms`. 