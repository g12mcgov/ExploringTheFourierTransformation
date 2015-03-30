#
#
#	Author: Grant McGovern
#	Date: 29 March 2015
#
#	CSC 222: Lab #3 
#	~ Roots of Unity & Fourier Transformation ~
#
#

import cmath 
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

class Complex(complex):
	def __repr__(self):
		rp = "%7.5f" % self.real if not self.pureImag() else ""
		ip = '%7.5fj' % self.imag if not self.pureReal() else ""
		conj = "" if (self.pureImag() or self.pureReal() or self.imag < 0.0) else "+"
		
		return "0.0" if (self.pureImag() and self.pureReal()) else rp+conj+ip

	def pureImag(self):
		return abs(self.real) < 0.000005 

	def pureReal(self):
		return abs(self.imag) < 0.000005 


# Computes n-roots of unity, using Complex number class
def croots(n):
	if n <= 0:
		return None
	else:
		return (Complex(cmath.rect(1, 2*k*cmath.pi/n)) for k in range(n))

# Computes c roots
def printCroots(n):
	for item in range(2, int(n+1)):
		print item, list(croots(item))

def computeFFT():
	N = 2
	T = (1.0 / 800.0)
	X = np.linspace(0.0, N*T, N)
	Y = np.sin(50.0 * 2.0*np.pi*X) + 0.5*np.sin(80.0 * 2.0*np.pi*X)

	yf = fft(Y)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	print yf

def computeDFT(x):
	x = np.asarray(x, dtype=float)
	N = x.shape[0]
	#print "N:", N
	n = np.arange(N)
	k = n.reshape((N, 1))
	M = np.exp(-2j * np.pi * k * n/N)

	return np.dot(M, x)

def testIfCorrect():
	pass

# Reads in text file, extracts data
def readInTextFile(fileName):
	print "hi"
	# with open(fileName) as fp:
	# 	signals = [line for line in fp]

	# 	print signals

		# return signals


if __name__ == "__main__":
	# printCroots(5)
	#computeFFT()
	matrix = computeDFT([4, 8, 16, 32])
	

	#print matrix
	print np.fft.ifft([4, 8, 16, 32])

	# print "Inverse:\n\n"
	# print np.linalg.inv(matrix)

	# readInTextFile("1Dsignal.txt")
	# print results

	x = np.random.random(1024)
	print np.allclose(computeDFT(x), np.fft.fft(x))


	#plt.plot(xf, 1.0/N * np.abs(yplot))
	
