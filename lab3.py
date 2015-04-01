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
import itertools
import numpy as np
import numpy.matlib
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

# def computeDFT(x):
# 	x = np.asarray(x, dtype=float)
# 	N = x.shape[0]
# 	#print "N:", N
# 	n = np.arange(N)
# 	k = n.reshape((N, 1))
# 	M = np.exp(-2j * np.pi * k * n/N)

# 	return np.dot(M, x)

def computeDFT(sequence):
	return np.fft.fft(sequence)

def computeInverseDFT(sequence):
	return np.fft.ifft(sequence)

def testIfCorrect(matrix):
	return np.allclose(matrix, np.fft.fft(matrix))

def plotSignal(signal):
	try:
		plt.plot(signal)
		plt.grid()
		plt.show()

	except AttributeError as err:
		raise err

def plotRootsOfUnity(roots):
	try:
		colors = itertools.cycle(['r', 'g', 'b', 'y'])

		plt.figure(figsize=(6,6))

		for root in roots:
			plt.arrow(0, 0, root.real, root.imag, ec=colors.next())

		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.show()

	except AttributeError as err:
		raise err

def computeIdentityMatrix(n):
	return np.matlib.identity(n)

# Reads in text file, extracts data
def readInTextFile(fileName):
	with open(fileName) as fp:
		return [line.strip('\n') for line in fp]

if __name__ == "__main__":
	# printCroots(5)'

	temp = Complex(1,1)
	temp2 = Complex(1, 1)
	
	print temp*temp2

	roots_2 = croots(2)
	roots_4 = croots(4)
	roots_8 = croots(8)

	


	signals = readInTextFile("1Dsignal.txt")

	g = [4, 8, 16, 32]

	ghat = computeDFT(g)
	g2 = computeInverseDFT(ghat)
	
	print "DFT:\n", ghat, "\n"
	# print "IDFT:\n", , "\n"
	for i in g2:
		print i.real

	print "DFT*IDFT:\n", np.dot(ghat, g2), "\n"
		
	plotSignal(signals)
	plotRootsOfUnity(roots)

	
