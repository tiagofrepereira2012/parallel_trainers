#!/usr/bin/env python

import xbob.db.mobio
import facereclib

mobio_image_directory = "[YOUR_MOBIO_IMAGE_DIRECTORY]"
mobio_annotation_directory = "[YOUR_MOBIO_ANNOTATION_DIRECTORY]"

database = facereclib.databases.DatabaseXBobZT(
    database = xbob.db.mobio.Database(),
    name = "mobio",
    original_directory = mobio_image_directory,
    original_extension = ".png",
    annotation_directory = mobio_annotation_directory,
    annotation_type = 'eyecenter',
    protocol = 'female',

    all_files_options = { 'gender' : 'female' },
    extractor_training_options = { 'gender' : 'female' },
    projector_training_options = { 'gender' : 'female' },
    enroller_training_options = { 'gender' : 'female' },
    z_probe_options = { 'gender' : 'female' }
)
