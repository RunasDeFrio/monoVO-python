import numpy as np 
import cv2
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab


PATH0 = '/home/runas/CodeProject/TestingData1/2/'

DRAW_STOP = False

from visual_odometry import PinholeCamera, VisualOdometry

k = 80
def coordToImage(x, z):
	return int(-k*x)+290, int(-k*z)+600-90

def resizeImg(img1):
	scale_percent = 50 # percent of original size
	width = int(img1.shape[1] * scale_percent / 100)
	height = int(img1.shape[0] * scale_percent / 100)
	dim = (width, height)
	# resize image
	return cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

while(True):

	fig = pylab.figure()
	Axes3D(fig)

	pylab.show()
	serial = PATH0 +'SerialRead.txt'
	cam_time = PATH0+'times'
	#cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
	cam = PinholeCamera(1920.0, 1080.0, 1954.453840, 1931.350674, 659.485149, 394.472409, 0.103832, -0.030339)
	vo = VisualOdometry(cam, serial, cam_time)

	traj = np.zeros((600,600,3), dtype=np.uint8)

	for i in traj:
		for j in i:
			j[0] = 255
			j[1] = 255
			j[2] = 255


	x0s = 0.0
	y0s = 0.0
	Xx = np.zeros((4))
	Xx[2]=0.5
	Xx[3]=1

	Zz = Xx

	_x = 0
	_y = 0
	_z = 0
	path = 0
	t0 = 0
	x_prev = 0
	y_prev = 0
	x10 = 0
	y10 = 0
	path0 = 'D:/DataSet/dataset/sequences/00/image_0/'
	path1 = PATH0 + 'image_0/'

	img_id = 0
	t_cam = vo.getCamTime(img_id)
	for i in range(451):#4541
		#all_time = time.time()
		#read_time = time.time()
		
		y_s, x_s, t_s = vo.getSerialData(i)

		y_s, x_s = 1*y_s, 1*x_s

		if(i == 0): 
			t0 = t_s

		t_s = t_s - t0

		cv2.circle(traj, (coordToImage(x_s, y_s)), 2, (0,255,0), 1)
		cv2.line(traj, (coordToImage(x0s, y0s)), (coordToImage(x_s, y_s)), (255,0,0))

		x0s = x_s
		y0s = y_s

		if (t_cam <= t_s and img_id < 17):

			img = cv2.imread(path1+str(img_id).zfill(6)+'.png', 0)
			#img = cv2.imread(path1+"TESTING_CAM"+str(img_id)+'.png', 0)
			#read_time = time.time() - read_time
			y_s1, x_s1, t_s1 = vo.getSerialData(i-1)

			x_true = (t_cam-t_s1)*(x_s - x_s1)/(t_s-t_s1) + x_s1
			y_true = (t_cam-t_s1)*(y_s - y_s1)/(t_s-t_s1) + y_s1

			cv2.circle(traj, (coordToImage(x_true, y_true)), 2, (255,0,0), 1)

			#absolute_scale = 1.414213562*np.sqrt((x_true - x_prev)*(x_true - x_prev) + (y_true - y_prev)*(y_true - y_prev))

			absolute_scale = 1.
			
			x_prev, y_prev = x_true, y_true

			vo.absolute_scale = absolute_scale
			#up_time = time.time()
			vo.update(img, img_id)
			#up_time = time.time() - up_time
			cur_t = vo.cur_t
			
			if(img_id > 0):
				x, y, z = cur_t[0]+x00, cur_t[1], cur_t[2]+z00
				Zz = vo.RT2P(vo.cur_R, cur_t).dot(Xx)
				Zz = Zz.reshape(4)
			else:
				x00, z00 = x_true, y_true
				x, y, z = x_true, x_true, y_true

			#true_x, true_y = coordToImage(vo.trueX, vo.trueZ)

			#print("error is:", x-x_true,z-y_true)
			#path = path + (((x-_x)**2+(_z-z)**2)**(1/2))
			#print("path:", path)


			#all_time = time.time() - all_time
			#print('all', all_time, 'read ', read_time/all_time*100, 'update ', up_time/all_time)

			for p in vo.point:
				cv2.circle(traj, coordToImage(p[0]+x00, p[2]+z00), 2, (255,0,0), -1)
			vo.point.clear()

			cv2.circle(traj, (coordToImage(x, z)), 2, (0,255,255), -1) #вывод положения камеры

			cv2.line(traj, (coordToImage(_x, _z)), (coordToImage(x, z)), (0,0,255)) #вывод пути камеры
			#cv2.line(traj, (coordToImage(x, z)), (coordToImage(Zz[0], Zz[2])), (128,0,255)) #вывод направления камеры
			#cv2.line(traj, (coordToImage(_x, _z)), (coordToImage(x, z)), (255,0,255))
			_x = x
			_y = y
			_z = z
			"""
                        #Вывод среднего значения
			k1 = 0.5
			x1 = k1*x + (1-k1)*x_true
			y1 = k1*z + (1-k1)*y_true
			cv2.line(traj, (coordToImage(x10, y10)), (coordToImage(x1, y1)), (0,255,255))
			cv2.circle(traj, (coordToImage(x1, y1)), 2, (0,0,255), 1)
                        
			x10, y10 = x1,y1
			"""
			cv2.rectangle(traj, (10, 20), (600, 60), (255,255,255), -1)
			text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
			cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, 8)

			if(img_id != 0):
				p2, p1 = vo.frames[-1].GetAllMatchKeyPoints()
				for i, pt1 in enumerate(p1):
					pt2 = p2[i]
					cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), (0,0,255))

				img1 = cv2.drawMatches(vo.frames[-2].frame, vo.frames[-2].keypoints, vo.frames[-1].frame, vo.frames[-1].keypoints, vo.matche, None, flags =  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
				
				
				cv2.imshow('IM', resizeImg(img))
				cv2.imshow('IMAGE', resizeImg(img1))
				DRAW_STOP = True
			
			img_id = img_id + 3
			if(img_id<17):
				t_cam = vo.getCamTime(img_id)
		cv2.imshow('Trajectory', traj)
		if(DRAW_STOP):
			cv2.waitKey(0)
		else:
			cv2.waitKey(1)
		DRAW_STOP = False
                

		

#	cv2.imwrite('map.png', traj)