import os
import zipfile


def read_zip_files(path_to_data):
    list_of_files = []
    zip_dict = {}
    zip_files = [x for x in os.listdir(path_to_data) if x.endswith(".zip")]
    for z in zip_files:
        archive = zipfile.ZipFile(os.path.join(path_to_data, z), "r")
        zip_dict[z] = archive.namelist()
        list_of_files += archive.namelist()
    return zip_dict, list_of_files