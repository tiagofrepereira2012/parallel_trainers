#!/usr/bin/env python

import xbob.db.cpqd
import facereclib

database = facereclib.databases.DatabaseXBobZT(

    database = xbob.db.cpqd.Database(),
    name = "cpqd",
    original_directory = "/home/biomodal/l/image_face_databases/cpqd/Baseline_20130821/images/",
    original_extension = ".jpg",
    annotation_directory = "/home/biomodal/l/image_face_databases/cpqd/Baseline_20130821/eyes",
    annotation_type = 'eyecenter',
    protocol = 'notebook_male'


)

