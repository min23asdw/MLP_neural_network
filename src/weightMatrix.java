public class weightMatrix {
    double[][] data;
    int rows,cols;

    /**
     * W ji weight form input neuron   i to j
     * @param rows j node
     * @param cols i node
     */


    public weightMatrix(int rows,int cols){
        data= new double[rows][cols];
        this.rows=rows;
        this.cols=cols;
        for(int j=0;j<rows;j++)
        {
            for(int i=0;i<cols;i++)
            {
                double ran = 0;
                while(ran == 0){
                    ran = Math.random()*2-1;
                    data[j][i]=ran;
                }
            }
        }
    }
}
