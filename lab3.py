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
from numpy.linalg import inv
from scipy.fftpack import fft
import matplotlib.pyplot as plt

class Complex(complex):
	def __repr__(self):
		rp = "%7.5f" % self.real if not self.pureImag() else ""
		ip = '%7.5fj' % self.imag if not self.pureReal() else ""
		conj = "" if (self.pureImag() or self.pureReal() or self.imag < 0.0) else "+"
		
		return "0.0" if (self.pureImag() and self.pureReal()) else (rp + conj + ip)

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
		name = 'signal.png' 
		plt.savefig(name)
		plt.show()

	except AttributeError as err:
		raise err

def plotFrequency(signal):
	try:
		plt.plot(signal)
		plt.grid()
		plt.show()

	except AttributeError as err:
		raise err

def plotRootsOfUnity(roots,i):
	try:
		colors = itertools.cycle(['r', 'g', 'b', 'y'])

		plt.figure(figsize=(6,6))

		for root in roots:
			plt.arrow(0, 0, root.real, root.imag, ec=colors.next())

		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)

		plt.axes()

		# Encomposes the graph in a circle
		circle = plt.Circle((0, 0), radius=1, fc='w')
		plt.gca().add_patch(circle)

		#Save the file as png
		name = 'roots_of_%d.png' % (i)
		plt.savefig("graphs/" + name)
		plt.show()


	except AttributeError as err:
		raise err

# A simple method to return an n x n identity matrix
def computeIdentityMatrix(n):
	return np.matlib.identity(n)

# Reads in text file, extracts data
def readInTextFile(fileName):
	with open(fileName) as fp:
		return [float(line.strip('\n')) for line in fp]

def computeF(n):
	return np.fft.fft(np.matlib.identity(n))

def computeIF(F):
	return inv(F)

# Checks if y and signals are equal
def checkIfEqual(y, signals):
	y1 = [float(int(element.real)) for element in y]

	if len(y) != len(signals): 
		raise Exception("Matrices not equal")
	else:
		for i, item in enumerate(signals):
			if (y1[i] - signals[i]) == 1 or -1:
				pass
			else:
				print "False"
				
		print "True"

def printcroots(roots,nth):
	print "\nRoots %s:\n"%(nth)
	
	for root in roots:
		negstring = " %-.5f + %-.5fi " % (root.real, root.imag)
		print negstring

if __name__ == "__main__":
	signals = readInTextFile("1Dsignal.txt")
	
	# Problem 1
	printcroots(croots(2),2)
	printcroots(croots(4),4)
	printcroots(croots(8),8)
	printcroots(croots(16),16)
	printcroots(croots(32),32)

	plotRootsOfUnity(croots(2),2)
	plotRootsOfUnity(croots(4),4)
	plotRootsOfUnity(croots(8),8)
	plotRootsOfUnity(croots(16),16)
	plotRootsOfUnity(croots(32),32)
	plotSignal(signals)

	# Problem 2
		
	# Computing F
	print "\n"
	print "F:"
	F = computeF(4)
	print F

	print "\n"
	print "F^-1:"
	F_inverse = computeIF(F)
	print F_inverse
	
	print "\n"
	print "Identity Matrix Check"
	print np.dot(F, F_inverse)

	# Computing F again step (b)
	F1D = computeF(1024)
	Finv1D = computeIF(computeF(1024))

	# Finding 'g^'
	g1D = np.dot(F1D, signals)

	print "\n"
	print g1D

	# Displays the frequency graph
	print "\nDisplaying Frequency Graph...\n"
	plotSignal(g1D)

	print "\nResult:\n"
	y = np.dot(g1D, Finv1D)

	# Sanity check to see if y is equal to signals after
	# all the operations have been performed.
	print "Matrices Equivalent: ", checkIfEqual(y, signals)



	
