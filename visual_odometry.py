import numpy as np 
import cv2
import math
from point_cloud import Frame

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500
Z = np.zeros((3))
Z[2] = 1
#Z[3]=1

MAX_FEATURES = 2000
GOOD_MATCH_PERCENT = 0.15

detector = cv2.ORB_create(MAX_FEATURES, 1.2, 8, 31, 0, 2, 0, 31)
#detector = cv2.ORB_create(400, 5, 5)
#detector = cv2.xfeatures2d.SIFT_create()

def Norm(X):
    X[0] = X[0]/X[3]
    X[1] = X[1]/X[3]
    X[2] = X[2]/X[3]
    X[3] = X[3]/X[3]
    return X

def Norm2(X):
    X[0] = X[0]/X[2]
    X[1] = X[1]/X[2]
    X[2] = X[2]/X[2]
    return X

class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = np.array([k1, k2, p1, p2])
		self.K = np.array([[fx,       0.0,          cx],
						  [0.0,        fy,          cy],
						  [0.0,        0.0,         1.0]])
		self.Kinv = np.linalg.inv(self.K)


class VisualOdometry:
	def __init__(self, cam, serial, camera):
		self.frame_stage = 0
		self.cam = cam
		self.frame = list()
		self.frames = list()
		self.cur_R = np.eye( 3 ,  dtype= 'float64') 
		self.cur_t = np.zeros((3,1))
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.detector = detector

		self.keypoints = list()
		self.descriptors = list()
		self.matches = list()
		self.matche = None

		#self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		self.point = []
		self.absolute_scale = 1.0

		with open(serial) as f:
			self.serialData = f.readlines()
		with open(camera) as f:
			self.cam_times = f.readlines()

	def getSerialData(self, n):  #specialized for KITTI odometry dataset
		ss = self.serialData[n].strip().split()
		t = float(ss[0])
		x = float(ss[5])/100
		y = float(ss[6])/100
		return x, y, t

	def getCamTime(self, n):  #specialized for KITTI odometry dataset
		ss = self.cam_times[n].strip().split()
		t = float(ss[0])
		return t

	def RT2P(self, R, T):
		T = T.reshape(3)
		return np.array(
              [[R[0][0],  R[0][1],    R[0][2],  T[0]],
               [R[1][0],  R[1][1],    R[1][2],  T[1]],
               [R[2][0],  R[2][1],    R[2][2],  T[2]],
               [      0, 	    0, 			0,     1]])

	def P2RT(self, P):
		R = np.array(
				[[P[0][0],  P[0][1],    P[0][2]],
				[P[1][0],  P[1][1],    P[1][2]],
				[P[2][0],  P[2][1],    P[2][2]]])

		T = np.array([P[0][3], P[1][3], P[2][3]])
			   
		return (R, T)

	def GetInvP(self, R, T):
		return self.RT2P(R.transpose(), -1*R.transpose().dot(T))


	def SortDistance(self, matches_all):
		min_d = 100
		for m in matches_all:
			if m.distance < min_d:
				min_d = m.distance

		matches = list()

		for m in matches_all:
			if(m.distance < 5*(min_d)):
				matches.append(m)
		return matches
		# Sort self.matches[-1] by score


	def DeleteEqual(self, matches_all):
		map_of_index = {}

		for m in matches_all:
			m2 = map_of_index.get(m.trainIdx)
			if(m2 != None):
				if(m.distance < m2.distance):
					map_of_index[m.trainIdx] = m
			else:
				map_of_index[m.trainIdx] = m
		matches = list(map_of_index.values()) 
		return matches
		# Sort self.matches[-1] by score



	def SortDelta(self, matches_all):
		matches = list()
		min_d = 100
		for m in matches_all:
			if(abs(self.frames[-2].keypoints[m.queryIdx].pt[0]-self.frames[-1].keypoints[m.trainIdx].pt[0])<(self.cam.width/3)):
				if(abs(self.frames[-2].keypoints[m.queryIdx].pt[1]-self.frames[-1].keypoints[m.trainIdx].pt[1])<(self.cam.height/5)):
					matches.append(m)
		return matches

	def SortLines(self, matches_all):
		def GetKB(x1, x2, y1, y2):
			dx = x1-x2
			dy = y1-y2
			

			if(dx == 0):
				k = 1000000
			else:
				k = dy/dx

			b = y1 - k*x1
			return (k, b)

		x0, y0 = None, None

		R = 100
		matches = list()
		Q = int(1000)
		traj = np.zeros((4*Q,4*Q,3), dtype=np.uint8)
		cv2.namedWindow('lines', cv2.WINDOW_NORMAL)

		sq = np.zeros((2,2), dtype = np.uint32)
		
		intersections = list()
		for i in range(len(matches_all)):
			m = matches_all[i]
			k0, b0 = GetKB(self.frames[-1].keypoints[m.trainIdx].pt[0], self.frames[-2].keypoints[m.queryIdx].pt[0], self.frames[-1].keypoints[m.trainIdx].pt[1], self.frames[-2].keypoints[m.queryIdx].pt[1])
			if(abs(b0) < 10*Q):
				cv2.line(traj, (int(Q-Q), int(-Q*k0+b0+Q)), (int(4*Q+Q), int(4*k0*Q+b0+Q)), (0,0,255), 1) #вывод пути камеры
			for j in range(i+1, len(matches_all)):
				k, b = GetKB(self.frames[-1].keypoints[matches_all[j].trainIdx].pt[0], self.frames[-2].keypoints[matches_all[j].queryIdx].pt[0], self.frames[-1].keypoints[matches_all[j].trainIdx].pt[1], self.frames[-2].keypoints[matches_all[j].queryIdx].pt[1])
				if(k0 == k):
					continue
				x = (b-b0)/(k0-k)
				y = k*x + b

				if(abs(x)> 6400 or abs(y)> 6400):
					continue
				cv2.circle(traj, (int(x+Q), int(y+Q)), 2, (0,255,0), 1)
				intersections.append((x, y))


		sq_size = 4*Q
		x0, y0 = -Q, -Q
		size_line = 0

		while (sq_size > 300 or (size_line/len(intersections)) > 0.6):
			for k in intersections:
				for i, row in enumerate(sq):
					for j, column in enumerate(row):
						xk = i*sq_size + sq_size/2 + x0
						yk = j*sq_size + sq_size/2 + y0
						if((abs(yk-k[1]) <= sq_size/2 )and (abs(xk-k[0]) <= sq_size/2)):
							sq[i, j] = sq[i, j] + 1
			i0, j0 = 0, 0
			for i, row in enumerate(sq):
				for j, column in enumerate(row):
					if(sq[i, j] > sq[i0, j0]):
						i0, j0 = i, j
						size_line = sq[i, j]
					text = str(sq[i, j])
					cv2.putText(traj, text, (int(i*sq_size + sq_size/2 + x0+Q),int(j*sq_size + sq_size/2 + y0+Q)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
			for i, row in enumerate(sq):
				for j, column in enumerate(row):
					sq[i, j] = 0
			x0, y0 = i0*sq_size+x0, j0*sq_size+y0
			cv2.rectangle(traj,(int(x0+Q),int(y0+Q)),(int(x0+sq_size+Q),int(y0+sq_size+Q)),(128,128,0),2)
			sq_size = sq_size / 2
		
		sq_size = sq_size * 2
		cv2.rectangle(traj,(int(x0+Q),int(y0+Q)),(int(x0+sq_size+Q),int(y0+sq_size+Q)),(0,255,0),3)
		for m in matches_all:
			k, b = GetKB(self.frames[-1].keypoints[m.trainIdx].pt[0], self.frames[-2].keypoints[m.queryIdx].pt[0], self.frames[-1].keypoints[m.trainIdx].pt[1], self.frames[-2].keypoints[m.queryIdx].pt[1])
			xk = sq_size/2 + x0
			yk = sq_size/2 + y0
			y = k*xk + b
			if(abs(yk-y) <= sq_size/2):
				matches.append(m)
				#if(abs(b) < 10*Q):
					#cv2.line(traj, (int(0), int(b)), (int(Q), int(k*Q+b)), (0,128,255), 3) #вывод пути камеры

		print('count - ',size_line, '::', len(intersections))
		cv2.imshow('lines', traj)
		cv2.waitKey(1)
		return matches

	def SortAndDelete(self):
		self.matches[-1].sort(key=lambda x: x.distance, reverse=False)
		# Remove not so good self.matches[-1]
		numGoodMatches = int(len(self.matches[-1]) * GOOD_MATCH_PERCENT)
		
		self.matches[-1] = self.matches[-1][:numGoodMatches]
		
	def orbTracking(self):

		key, descriptor = self.detector.detectAndCompute(self.frame, None)

		self.frames.append(Frame(self.frame, key, descriptor))

		matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
		
		matches_all = matcher.match(self.frames[-2].descriptors, self.frames[-1].descriptors, None)
		
		matches_all = self.SortDelta(matches_all)
		matches_all = self.SortDistance(matches_all)
		matches_all = self.DeleteEqual(matches_all)

		#mat = self.SortLines(matches_all)

		#if(len(mat) > 0):
		#	matches_all = mat
		self.matche = matches_all
		self.frames[-1].mapingKeyPointsInFrame(matches_all, self.frames[-2])
				
	def LinearLSTriangulation(self, x, P, x1, P1):
		A  = np.zeros((4,4))
		A [0] = (x[0] * P[2] - P[0])
		A [1] = (x[1] * P[2] - P[1])
		A [2] = (x1[0] * P1[2] - P1[0])
		A [3] = (x1[1] * P1[2] - P1[1])

		B = np.zeros((4,1))
		B[0,0] = 0.0001
		B[1,0] = 0.0001 
		B[2,0] = 0.0001 
		B[3,0] = 0.0001
		u, s, vh = np.linalg.svd(A)
		
		#X = np.linalg.solve(A, B).reshape(4)
		X = vh[3].transpose()
		
		Norm(X)
		#Norm(X1)
		#print(X-X1, '\n')
		X_ = np.zeros((3), dtype = 'float32')
		X_[0] = X[0]
		X_[1] = X[1]
		X_[2] = X[2]
		return X_

	def Triang(self):
		frame2 = self.frames[-1]
		for i in frame2.triang_point:
			j, frame1 = frame2.key_map[i]
			
			R1, t1 = frame1.R, frame1.T
			R2, t2 = frame2.R, frame2.T
			P1, P2 = frame1.GetP(), frame2.GetP()
			
			u1 = np.array([[frame1.keypoints[j].pt[0]], [frame1.keypoints[j].pt[1]],[1.0]])
			u2 = np.array([[frame2.keypoints[i].pt[0]], [frame2.keypoints[i].pt[1]],[1.0]])

			x1 = (self.cam.Kinv.dot(u1))
			x2 = (self.cam.Kinv.dot(u2))

			ray1 = np.transpose(R1).dot(x1).reshape(3)
			ray2 = np.transpose(R2).dot(x2).reshape(3)	
			
			cosParallaxRays = ray1.dot(ray2)/(np.linalg.norm(ray1)*np.linalg.norm(ray2))
			
			X = self.LinearLSTriangulation(x1, P1, x2, P2)

			#Check triangulate z
			z1 = R1.dot(X) + t1.reshape(3)
			z2 = R2.dot(X) + t2.reshape(3)
			
			u10 = Norm2(self.cam.K.dot(R1.dot(X) + t1.reshape(3)))
			u20 = Norm2(self.cam.K.dot(R2.dot(X) + t2.reshape(3)))

			err1 =  ((u1[0] - u10[0])**2 + (u1[1] - u10[1])**2)**(1/2)
			err2 =  ((u2[0] - u20[0])**2 + (u2[1] - u20[1])**2)**(1/2)
			print(X)
			if(z1[2]>0 and z2[2]>0 and cosParallaxRays>0 and cosParallaxRays<0.9998 and err1 < 10 and err2 < 10):# and X[1]> P0[1,3]): X[2] > P0[2,3]
				self.point.append(X)
				self.frames[-1].SetSpacePoint(i, X, True)
			else:
				self.frames[-1].SetSpacePoint(i, X, False)
		
		print('Triang point:', len(self.point), ':', len(self.frames[-1].triang_point))
		self.frames[-1].triang_point.clear()

		spaces, points = self.frames[-1].GetVisibleSpacePoint()
		#_, R, t, A = cv2.solvePnPRansac(spaces, points, self.cam.K, self.cam.d)
		#R, j = cv2.Rodrigues(R)

		#print('P2P solve:\nR:\n',R.transpose(),'P2P solve:\nt:\n', -1 * R.transpose().dot(t),'Track:\nR:\n', self.cur_R,'Track:\nt:\n', self.cur_t)
		#print("HERE")





	def TrackPose(self):
		self.px_ref, self.px_cur = self.frames[-1].GetAllMatchKeyPoints()
		E, m = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)

		R1, R2, t = cv2.decomposeEssentialMat(E)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)

		if(R.dot(Z)[2]<0):
			if(np.sum(R-R1) < 0.0000000001): #магическое сравнение между собой, не спрашивайте
				R = R2
			else:
				R = R1

		self.ref_t = self.cur_t
		self.ref_R = self.cur_R
		
		#scale = self.absolute_scale/((t[0]**2+t[2]**2)**(1/2))

		self.cur_t = self.cur_t + self.cur_R.dot(t)

		t__ = self.cur_t-self.ref_t
		if(t__[2]<0):
			t = -t
			self.cur_t = self.ref_t + self.absolute_scale*self.cur_R.dot(t) 
		
		self.cur_R = R.dot(self.cur_R)

		self.frames[-1].R = self.cur_R.transpose()
		self.frames[-1].T = -1 * self.cur_R.transpose().dot(self.cur_t)

	def processFirstFrame(self):

		key, descriptor = self.detector.detectAndCompute(self.frame, None)

		self.frames.append(Frame(self.frame, key, descriptor))
		
		self.frame_stage = STAGE_SECOND_FRAME

	def processFrame(self):
		self.orbTracking()
		spaces, points = self.frames[-1].GetVisibleSpacePoint()
	
		_, R, t, A = cv2.solvePnPRansac(spaces, points, self.cam.K, self.cam.d)
		R, j = cv2.Rodrigues(R)

		t = -1 * R.transpose().dot(t)
		R = R.transpose()

		self.ref_t = self.cur_t
		self.ref_R = self.cur_R
		
		self.cur_t = t
		self.cur_R = R

		self.frames[-1].R = self.cur_R.transpose()
		self.frames[-1].T = -1 * self.cur_R.transpose().dot(self.cur_t)
		self.Triang()

	def update(self, img, frame_id):
		assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
		self.frame = img

		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame()
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.orbTracking()
		
			self.TrackPose()

			self.Triang()
			self.frame_stage = STAGE_DEFAULT_FRAME

		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
