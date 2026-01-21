#include <Copper/Copper.h>

extern "C"
{
	bool Cu_Initialize() {
		cudaFree(0);

		ForceGPUPerformance();
		return true;
	}

	void Cu_Shutdown() {
		printf("Shutdown\n");
	}
}
