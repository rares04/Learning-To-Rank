# Simple script that saves a backup for a given file

def make_backup(in_file_name, out_file_name):
    in_file = open(in_file_name, "r")
    content = in_file.readlines()
    in_file.close()

    out_file = open(out_file_name, "w")
    out_file.write("".join(content))
    out_file.close()