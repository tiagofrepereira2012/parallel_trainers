#!/usr/bin/env python

import xbob.db.cpqd
import facereclib

database = facereclib.databases.DatabaseXBobZT(

    database = xbob.db.cpqd.Database(),
    name = "cpqd",
    original_directory = "/home/biomodal/l/tiago/views/facereclib/EXPERIMENTS/cpqd/cpqd_n2s_male/baselines/isv/features/",
    original_extension = ".hdf5",
    annotation_directory = "",
    annotation_type = 'eyecenter',
    protocol = 'n2s_male'



)

