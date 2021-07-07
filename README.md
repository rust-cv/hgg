# hgg

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/hgg.svg
[cl]: https://crates.io/crates/hgg/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/hgg/badge.svg
[dl]: https://docs.rs/hgg/

[lo]: https://tokei.rs/b1/github/rust-cv/hgg?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/hgg/workflows/tests/badge.svg
[bll]: https://github.com/rust-cv/hgg/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/hgg/workflows/no-std/badge.svg

Hierarchical Greedy Graph

## Recall Curve

To generate the `akaze` file used in the `recall_akaze` example, you first need a large number of unique images. It is recommended that these images come from a vSLAM dataset (TUM, KITTI, etc) if you want to test the performance in that particular setting.

For example, to get a chunk of the KITTI dataset, you can go to http://www.cvlibs.net/datasets/kitti/raw_data.php and download `2011_09_29_drive_0071` using the `[link unsynced+unrectified data]`. [Here](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0071/2011_09_29_drive_0071_extract.zip) is a link, though it may break. Extract the zip into a directory.

You will need to first install `docopt`, `progress`, and `opencv-contrib-python` (recommended) or `opencv-python` (use if you are worried about licensing issues). Next, you need to extract the features. `generate.py` is provided in this repository to do this. Here is an example:

```bash
echo ~/2011_09_29/2011_09_29_drive_0071_extract/image_*/data/* | xargs python generate.py akaze > akaze
```


It will provide a progress bar for each batch passed to it, but due to argument limit limitations it cant give you progress for the whole dataset. This will generate a file called `akaze` which contains many `61` byte AKAZE binary descriptors. You can do this with any dataset of images, but you may need to adjust the constant `HIGHEST_POWER_SEARCH_SPACE` in the `recall_akaze` example to a lower value if you do not have enough features in your file, which it will let you know when you run it.

To run the benchmark, it is recommended to first source `runrecall.sh`:

```bash
source runrecall.sh
```

This only needs to be done once per bash session, and will add the runrecall command to your bash session. Now run the following:

```bash
runrecall recall
```

This will begin running the program. You will see the following start to appear into `recall.csv`:

```csv
recall,search_size,knn,num_queries,seconds_per_query,queries_per_second,strategy
1.0,1,1,32768,1.58e-7,6329113.924050633,zero
1.0,1,1,32768,2.01e-7,4975124.378109452,regular
1.0,1,1,32768,7.8e-8,12820512.820512822,wide
1.0,1,2,32768,1.61e-7,6211180.124223603,zero
```

This file is populated as the benchmark runs. You can terminate the benchmark at any time with Ctrl+C, and the results computed so far will remain in this CSV file. It can be imported into a spreadsheet processor to generate graphs.
