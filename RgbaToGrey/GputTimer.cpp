#include "GPUTimer.h"

#include <iostream>


GPUTimer::GPUTimer() : m_Started(false), m_Stopped(false)
{
	cudaEventCreate(&m_StartEvent);
	cudaEventCreate(&m_StopEvent);
}

GPUTimer::~GPUTimer()
{
	cudaEventDestroy(m_StartEvent);
	cudaEventDestroy(m_StopEvent);
}

// Start event timer
void GPUTimer::start_timer(cudaStream_t stream_id)
{
	cudaEventRecord(m_StartEvent, stream_id);
	m_Started = true;
	m_Stopped = false;
}

// End event timer
void GPUTimer::stop_timer(cudaStream_t stream_id)
{
	if (!m_Started)
	{
		std::cout << "Timer hasn't started yet. Please call start_timer() before!" << std::endl;
		return;
	}
	cudaEventRecord(m_StopEvent, stream_id);
	m_Started = false;
	m_Stopped = true;
}

// Print elapsed time
void GPUTimer::print_elapsed_time()
{
	if (!m_Stopped)
	{
		std::cout << "Timer hasn't stopped yet. Please call stop_timer() before!" << std::endl;
		return;
	}
	cudaEventSynchronize(m_StopEvent);
	float elapsed_time = 0.0f;
	cudaEventElapsedTime(&elapsed_time, m_StartEvent, m_StopEvent);

	std::cout << "Elapsed GPU Time: " << elapsed_time << " msec" << std::endl;
}