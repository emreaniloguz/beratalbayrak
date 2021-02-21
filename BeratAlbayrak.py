import cv2

find_berat_albayrak = cv2.imread('berat2.jpg')
find_berat_albayrak = cv2.resize(find_berat_albayrak,(1000,1000))
cv2.imshow('Where is BERAT ALBAYRAK', find_berat_albayrak)
cv2.waitKey(0)

gray = cv2.cvtColor(find_berat_albayrak, cv2.COLOR_BGR2GRAY)

template_match= cv2.imread('beratalbayrak.jpeg', 0)

template_match = template_match[:, 175:210]
cv2.imshow('Template Matching', template_match)

result = cv2.matchTemplate(gray, template_match, cv2.TM_CCOEFF)
cv2.imshow("Result",result)
cv2.waitKey(0)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0]+ 50, top_left[1]+50)
cv2.rectangle(find_berat_albayrak, top_left, bottom_right, (0,0,255), 5)
#cv2.imshow('Where is BERAT ALBAYRAK', find_berat_albayrak)
cv2.imwrite('berat.jpeg', find_berat_albayrak)