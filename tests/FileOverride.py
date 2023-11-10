
captured_file_data = {}
class FileManager:
    def __init__(self, file_name: str) -> None:
        global captured_file_data
        self.file_name = file_name
        captured_file_data[file_name] = ""
    def write(self, str):
        global captured_file_data
        captured_file_data[self.file_name] += str
    def close(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

open_original = open
def open_override(f_name, *args, **kwargs):
    return_file = open_original(f_name, *args, **kwargs)
    if len(args) > 0:
        if args[0] == 'w':
            return_file = FileManager(f_name)
    elif 'mode' in kwargs:
        if kwargs['mode'][0] == 'w':
            return_file = FileManager(f_name)
    return return_file
# open_old = open
# open = my_open2