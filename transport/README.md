# TRANSPORT SCRIPTS

Make sure the tasks are done in this order, and that all of your scripts are in the same folder, for there is a (manual) pipeline of files & scripts that requires it! 

## Getting information on the existing files

The `list_jpg.bat` and `txt_list.jpg` go recursively through a folder you specify(at the top of the batch files) and all its subfolders to retrieve the lists (with full links) of the images and labels and store them into `jpg_list.txt` and `txt_list.txt` respectively.

## Moving files into a folder

The `transport_images.bat` and `transport_labels.bat` will read the `jpg_list.txt` and `txt_list.txt` files respectively, and line by line (reading the full links of files) and copy them into the two newly created folders titled `images` and `labels`.

## Getting the lists of files with relative links

Since the `txt_list.txt` and `jpg_list.txt` store the whole links, rather than relative links, we gotta convert them to the relative links:
1. We will first get the clean names of files, by running the `slicer_list4.bat` (it's for images) and `slicer_label.bat`. This will create `filenames.txt` and `labelnames.txt` respectively.
2. Then we run `image_locations.bat` and `label_locations.bat` which take in the `filenames.txt` and `labelnames.txt` and just appends the folder in which images and labels are (you can specify which in the for loop after the echo). This will then create the `images_locate.txt` and `labels_locate.txt` which hold the lists of relative links to images and labels.
