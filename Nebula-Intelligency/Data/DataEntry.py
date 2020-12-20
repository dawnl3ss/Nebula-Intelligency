class DataEntry(object):

    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def set_data(self, new_data):
        self.data = new_data