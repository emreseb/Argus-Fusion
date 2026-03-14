import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("input_folder", output="output",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, group=None,
    formats=None, move=False, shuffle=True) # default values