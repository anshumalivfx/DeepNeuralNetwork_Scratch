#pragma once


#include<vector>

#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"
#include <Eigen/Core>

class Layer
{
protected:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	const int m_inputSize;
	const int m_outputSize;
	const int m_batchSize;
public:
	Layer(const int inputSize, const int outputSize) :
		m_inputSize(inputSize), m_outputSize(outputSize) {

	}
	
		
	virtual ~Layer();

	int inputSize() const { return m_inputSize; }
	int outputSize() const { return m_outputSize; }

	virtual void init(const Scalar mu, const Scalar sigma, RNG& rng) = 0;
	virtual void forward(const Matrix& input, Matrix& output) = 0;

	virtual const Matrix& getOutput() const = 0;

	virtual void backward(const Matrix& pre_layer_output, const Matrix& next_layer_output) = 0;
	virtual const Matrix& backprop_data()const = 0;

Layer::Layer()
{
}

Layer::~Layer()
{
}

