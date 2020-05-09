
class bbox:
    def __init__(self, left, top, right, bottom, key='', value=''):
        self.key = key
        self.value = value
        self.xmin = left
        self.ymin = top
        self.width = right-left
        self.height = bottom-top
        self.xmax = right
        self.ymax = bottom
        self.line_str=''

    def rescale(self, scale_x, scale_y):
        self.xmin= self.xmin*scale_x
        self.ymin=self.ymin*scale_y
        self.width=self.width*scale_x
        self.height=self.height*scale_y
        self.xmax=self.xmax*scale_x
        self.ymax=self.ymax*scale_y
        return

    def offset(self, x = 0, y = 0):
        self.xmin=self.xmin-x
        self.ymin=self.ymin-y
        return self.line_str

    def asign_value(self, value):
        self.value=value

    def to_line_str(self):
        self.line_str=self.value+" "+str(int(self.xmin))+" "+str(int(self.ymin))+" "+str(int(self.width))+" "+str(int(self.height))

    def export_to_json(self):
        retjson = {
            "value": self.value,
            "x1" : round(self.xmin,6),
            "x2" : round(self.xmax,6),
            "y1" : round(self.ymin,6),
            "y2": round(self.ymax,6)
        }

        return retjson