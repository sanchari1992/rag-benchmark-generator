class BaseAdapter:

    def inspect(self):
        raise NotImplementedError()

    def query(self, question):
        raise NotImplementedError()

    def export_chunks(self):
        raise NotImplementedError()