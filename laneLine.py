import cv2
import numpy as np
import matplotlib.pyplot as plt


def slidingWsearch(img):

	nwindows = 10
	margin = 100
	minpix = 50
	### histogram of lower half of the image
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

	out_img = np.dstack((img, img, img))*255
### left and right base
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
### window height
	window_height = np.int(img.shape[0]/nwindows)
### searching non zero pixels
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	leftx_current = leftx_base
	rightx_current = rightx_base

	left_lane_inds = []
	right_lane_inds = []
	for window in range(nwindows):
        ###boundries identification
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
        ### drawing window
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        ###non zero pixel identification
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        ### appending pixels
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
        ###centering of the next window
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)


	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

    ###2nd polinomial fitting
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

	return left_fit, right_fit, out_img

### birds eye view
def warping(img):
    points = [(275,230), (370,230), (150,315), (560,315)]
    src = np.float32(points)
    dst = np.float32([
        (0,0),
        (640, 0),
        (0, 360),
        (640, 360)
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img, M, (640, 360), flags=cv2.INTER_LINEAR)

    return warped, M
### unwarping the image
def unwarp(img):
	points = [(275,230), (370,230), (150,315), (560,315)]
	src = np.float32(points)
	dst = np.float32([
        (0,0),
        (640, 0),
        (0, 360),
        (640, 360)
    ])

	Minv = cv2.getPerspectiveTransform(dst,src)
	warpedB = cv2.warpPerspective(img, Minv, (640, 360), flags=cv2.INTER_LINEAR)
	return warpedB, Minv
### image thresholding
def treshold(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	_,thresholded = cv2.threshold(img,160,255,cv2.THRESH_BINARY)

	return thresholded
### final image
def finalImage(img,thr,left_fit,right_fit):
	warp_zero = np.zeros_like(thr).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	ploty = np.linspace(0, thr.shape[0]-1, thr.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
	cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)

	warpedB, Minv = unwarp(color_warp)
	result = cv2.addWeighted(img, 1, warpedB, 0.8, 0)
	return result

cap =cv2.VideoCapture("test_video(1).mp4")
### video writing kodec
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('outputBW.avi',fourcc, 10.0, (640,360))
while(cap.isOpened()):
	_, frame = cap.read()

	img, M = warping(frame)
	thr = treshold(img)
	left_fit, right_fit, out_img = slidingWsearch(thr)
	final = finalImage(frame,thr,left_fit,right_fit)
	cv2.imshow('e',thr)
	cv2.imshow('w', final)
	### writing and saving video file
	#out.write(thr)
	#out.write(final)

	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
#out.release()
cv2.destroyAllWindows()
