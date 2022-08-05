# REEF Sample for DISB

# Download REEF
```bash
$ git submodule init --update
```

# Build REEF

see https://github.com/SJTU-IPADS/reef/blob/main/INSTALL.md

# Build DISB and REEF

```bash
# in disb

$ cmake -DSAMPLE_REEF=ON -B build
$ cd build
$ make -j$(nproc)
```