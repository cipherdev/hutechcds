

import numpy as np


class PID :
	def __init__(self,kp,ki,kd):
		self.p_error = 0
		self.i_error=0
		self.d_error=0
		self.Kp_pid=kp
		self.Ki_pid=ki
		self.Kd_pid=kd
		self.first_cte =True 
		self.prev_cte =0
  		#self.steer = 0
		self.sum_cte = 0
  		

	def update_error(self,cte):
		"""
			Truyen vao center track error

		"""
		if(self.first_cte):
			self.prev_cte = cte
			self.first_cte = False
		#------------------------
		# print(self.prev_cte,"prev_cte:")
		self.p_error=cte
		self.d_error=cte-self.prev_cte
		self.i_error=self.sum_cte
		self.prev_cte=cte
		self.sum_cte+=cte
	



	def total_error(self):
		"""
			Truyen lai tat ca cac loi

		"""
		# print(self.p_error)
		# print(self.d_error)
		# print(self.i_error)

		# print(self.Kp_pid)
		# print(self.Kd_pid)
		# print(self.Ki_pid)
		steer = -1*(self.Kp_pid*self.p_error+self.Kd_pid*self.d_error+self.Ki_pid*self.i_error)
		# if(steer>1):
		# 	steer=1
		# if(steer<-1):
		# 	steer=-1
		return steer



def main():
	kp=0.2
	ki=0.3
	kd=0.4
	pid=PID(kp,ki,kd)
	cte=9
	pid.update_error(cte)
	steer=pid.total_error()
	print(steer,"1 time")

	cte=-8
	pid.update_error(cte)
	steer=pid.total_error()
	print(steer,"2 time")


if __name__ == '__main__':
	main()

