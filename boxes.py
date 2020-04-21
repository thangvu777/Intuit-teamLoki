import fitz
import json

def SortBlocks(blocks):
    '''
    Sort the blocks of a TextPage in ascending vertical pixel order,
    then in ascending horizontal pixel order.
    This should sequence the text in a more readable form, at least by
    convention of the Western hemisphere: from top-left to bottom-right.
    If you need something else, change the sortkey variable accordingly ...
    '''

    sblocks = []
    for b in blocks:
        x0 = str(int(b["bbox"][0]+0.99999)).rjust(4,"0") # x coord in pixels
        y0 = str(int(b["bbox"][1]+0.99999)).rjust(4,"0") # y coord in pixels
        sortkey = y0 + x0                                # = "yx"
        sblocks.append([sortkey, b])
    sblocks.sort()
    return [b[1] for b in sblocks] # return sorted list of blocks

def SortLines(lines):
    ''' Sort the lines of a block in ascending vertical direction. See comment
    in SortBlocks function.
    '''
    slines = []
    for l in lines:
        y0 = str(int(l["bbox"][1] + 0.99999)).rjust(4,"0")
        slines.append([y0, l])
    slines.sort()
    return [l[1] for l in slines]

def SortSpans(spans):
    ''' Sort the spans of a line in ascending horizontal direction. See comment
    in SortBlocks function.
    '''
    sspans = []
    for s in spans:
        x0 = str(int(s["bbox"][0] + 0.99999)).rjust(4,"0")
        sspans.append([x0, s])
    sspans.sort()
    return [s[1] for s in sspans]  

def formatCoordinate(x):
	return str(int(x[0])) + ',' + str(int(x[1]))

def read_pdf(dir:str):
  doc = fitz.open(dir)
  page = doc[0]
  tp = page.getTextPage()
  # 'html', 'blocks', 'json', 'rawdict', ''
  text = page.getText('words')
  # page_dict = json.loads(text)
  # blocks = page_dict['blocks']
  # blocks = SortBlocks(blocks)
  for word in text: 
  	top_left = (word[0],word[1])
  	top_right = (word[2],word[1])
  	bottom_right = (word[2],word[3])
  	bottom_left = (word[0],word[3])
  	outputString = formatCoordinate(top_left) + ',' + formatCoordinate(top_right) + ',' + formatCoordinate(bottom_right) + ',' + formatCoordinate(bottom_left) + ',' + word[4]
  	print(outputString)
    #print(word[0],word[1],word[2],word[3],word[4])




  

if __name__ == '__main__':
    read_pdf('/Users/Taaha/Desktop/W2_XL_input_clean_1000.pdf')