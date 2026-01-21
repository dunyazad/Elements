#include <Copper/Copper.h>

extern "C"
{
	bool Cu_Initialize() {
		ForceGPUPerformance();
		return true;
	}

	void Cu_Shutdown() {
		printf("Shutdown\n");
	}
}
