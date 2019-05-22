import numpy as np 
import cv2
import math
from point_cloud import *
from core import *
from detecting import Detector

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

Z = np.zeros((3))
Z[2] = 1
#Z[3]=1

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
		self.detector = Detector(self.frames, cam)
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
			
			X = LinearLSTriangulation(x1, P1, x2, P2)

			#Check triangulate z
			z1 = R1.dot(X) + t1.reshape(3)
			z2 = R2.dot(X) + t2.reshape(3)
			
			u10 = Norm2(self.cam.K.dot(R1.dot(X) + t1.reshape(3)))
			u20 = Norm2(self.cam.K.dot(R2.dot(X) + t2.reshape(3)))

			err1 =  ((u1[0] - u10[0])**2 + (u1[1] - u10[1])**2)**(1/2)
			err2 =  ((u2[0] - u20[0])**2 + (u2[1] - u20[1])**2)**(1/2)
			#print(X)
			if(z1[2]>0 and z2[2]>0 and cosParallaxRays>0 and cosParallaxRays<0.9998 and err1 < 10 and err2 < 10):# and X[1]> P0[1,3]): X[2] > P0[2,3]
				self.point.append(X)
				self.frames[-1].SetSpacePoint(i, X, True)
			else:
				self.frames[-1].SetSpacePoint(i, X, False)
		
		print('Triang point:', len(self.point), ':', len(self.frames[-1].triang_point))
		self.frames[-1].triang_point.clear()

		#spaces, points, index = self.frames[-1].GetVisibleSpacePoint()
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

		key, descriptor = self.detector.Detect(self.frame)

		self.frames.append(Frame(self.frame, key, descriptor))
		
		self.frame_stage = STAGE_SECOND_FRAME

	def processFrame(self):

		self.matche = self.detector.orbTracking(self.frame)
		spaces, points, index = self.frames[-1].GetVisibleSpacePoint()
	
		_, R, t, A = cv2.solvePnPRansac(spaces, points, self.cam.K, self.cam.d)
		
		i = 0
		j = 0
		while(i < len(points)):
			if(j == A.shape[0]):
				self.frames[-1].PopKey(index[i])
			elif(i == A[j][0]):
				j += 1
			else:
				self.frames[-1].PopKey(index[i])
			i += 1

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
			self.matche = self.detector.orbTracking(self.frame)
		
			self.TrackPose()

			self.Triang()
			self.frame_stage = STAGE_DEFAULT_FRAME

		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
