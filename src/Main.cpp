//
//  main.cpp
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include <iostream>

#include "config.h"
#include "Cube.h"
#include "Connector.h"

/**
 * Input:
 BATCH 0 DEPTH 0
 a1 d1 g1
 b1 e1 h1
 c1 f1 i1
 BATCH 0 DEPTH 1
 a1' d1' g1'
 b1' e1' h1'
 c1' f1' i1'
 BATCH 1 DEPTH 0
 a2 d2 g2
 b2 e2 h2
 c2 f2 i2
 BATCH 1 DEPTH 1
 a2' d2' g2'
 b2' e2' h2'
 c2' f2' i2'
 *
 * Expect output with Kernel size 3x3:
 *
 BATCH 0 DEPTH 0
 a1 d1 b1 e1 a1' d1' b1' e1'
 d1 g1 e1 h1 d1' g1' e1' h1'
 b1 e1 c1 f1 b1' e1' c1' f1'
 e1 h1 f1 i1 e1' h1' f1' i1'
 a2 d2 b2 e2 a2' d2' b2' e2'
 d2 g2 e2 h2 d2' g2' e2' h2'
 b2 e2 c2 f2 b2' e2' c2' f2'
 e2 h2 f2 i2 e2' h2' f2' i2'
 *
 **/
void TEST_LOWERING(){
    Cube<DataType_String, Layout_RCDB> cube1(3, 3, 2, 2);
    
    Cube<DataType_String, Layout_RCDB> cube2(2*2*2, (3-2+1)*(3-2+1)*2, 1, 1);
    
    LoweringConfig lconfig;
    lconfig.kernel_size = 2;
    
    Connector<DataType_String, Layout_RCDB, DataType_String, Layout_RCDB, Connector_Lowering_R1C1>
        connector(&cube1, &cube2, &lconfig);
    
    size_t ct = 0;
    cube1.p_data[ct++] = "a1"; cube1.p_data[ct++] = "b1";
    cube1.p_data[ct++] = "c1"; cube1.p_data[ct++] = "d1";
    cube1.p_data[ct++] = "e1"; cube1.p_data[ct++] = "f1";
    cube1.p_data[ct++] = "g1"; cube1.p_data[ct++] = "h1";
    cube1.p_data[ct++] = "i1";
    
    cube1.p_data[ct++] = "a2"; cube1.p_data[ct++] = "b2";
    cube1.p_data[ct++] = "c2"; cube1.p_data[ct++] = "d2";
    cube1.p_data[ct++] = "e2"; cube1.p_data[ct++] = "f2";
    cube1.p_data[ct++] = "g2"; cube1.p_data[ct++] = "h2";
    cube1.p_data[ct++] = "i2";
    
    cube1.p_data[ct++] = "a1'"; cube1.p_data[ct++] = "b1'";
    cube1.p_data[ct++] = "c1'"; cube1.p_data[ct++] = "d1'";
    cube1.p_data[ct++] = "e1'"; cube1.p_data[ct++] = "f1'";
    cube1.p_data[ct++] = "g1'"; cube1.p_data[ct++] = "h1'";
    cube1.p_data[ct++] = "i1'";
    
    cube1.p_data[ct++] = "a2'"; cube1.p_data[ct++] = "b2'";
    cube1.p_data[ct++] = "c2'"; cube1.p_data[ct++] = "d2'";
    cube1.p_data[ct++] = "e2'"; cube1.p_data[ct++] = "f2'";
    cube1.p_data[ct++] = "g2'"; cube1.p_data[ct++] = "h2'";
    cube1.p_data[ct++] = "i2'";
    
    connector.transfer(&cube1, &cube2);
    
    cube2.logical_print();
    
    connector.report_last_transfer.print();
    connector.report_history.print();
    connector.transfer(&cube1, &cube2);
    connector.report_last_transfer.print();
    connector.report_history.print();
}

void TEST_TIMER(){
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;
    
    Cube<DataType_SFFloat, Layout_RCDB> cube1(64, 64, 96, 12);
    
    Cube<DataType_SFFloat, Layout_RCDB> cube2(lconfig.kernel_size*lconfig.kernel_size*96,
                        (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);
    
    Connector<DataType_SFFloat, Layout_RCDB, DataType_SFFloat, Layout_RCDB, Connector_Lowering_R1C1>
    connector(&cube1, &cube2, &lconfig);
    
    connector.transfer(&cube1, &cube2);
    
    connector.report_last_transfer.print();
    connector.report_history.print();
    connector.transfer(&cube1, &cube2);
    connector.report_last_transfer.print();
    connector.report_history.print();

}


int main(int argc, const char * argv[]) {
    
    //TEST_LOWERING();

    TEST_TIMER();
    
    /*
    Cube<DataType_SFFloat, Layout_RCDB> cube1(
        10, 10, 10, 10
    );
    
    Cube<DataType_SFFloat, Layout_RCDB> cube2(
        3*3*10, (10-3+1)*(10-3+1)*10, 1, 1
    );
    */
    
    //DataType_SFFloat * data = cube.logical_get(1,2,3,4);
    //*data = 5;
    //data = cube.logical_get(1,2,3,4);
    //std::cout << *data << std::endl;
    //cube.physical_get_RCslice(3,4);
    
    /*
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;
    
    Connector<DataType_SFFloat, Layout_RCDB,
              DataType_SFFloat, Layout_RCDB,
              Connector_Lowering_R1C1>connector(&cube1, &cube2, &lconfig);
    
    connector.transfer(&cube1, &cube2);
    */
     
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}
