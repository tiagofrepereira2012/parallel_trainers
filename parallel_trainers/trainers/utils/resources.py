#!/usr/bin/env python
#Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Fri Oct 05 12:08:00 CEST 2012

import pkg_resources

"""
Utilitary functions to access the database resources
"""

def get_available_resources(keyword):
  """
  """
  databases = []
  for entrypoint in pkg_resources.iter_entry_points(keyword):
    databases.append(entrypoint.name)

  return databases


def load_resource(keyword,resource_name):
  """
  """

  for entrypoint in pkg_resources.iter_entry_points(keyword):
    if(resource_name == entrypoint.name):
      plugin = entrypoint.load()
      return plugin

  return None
