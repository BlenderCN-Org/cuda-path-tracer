#include "MainWindow.h"
int main(int argc,char ** argv){
	MainWindow mw;
	if(argc>1) {
		mw.loadFile(argv[1]);
		mw.mainLoop();
	}
return 0;
}
