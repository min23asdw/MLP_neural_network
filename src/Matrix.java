import java.util.Random;

public class Matrix {
    double[][] data;
    int rows,cols;

    /**
     * W ji weight form input neuron   i to j
     * @param rows j node
     * @param cols i node
     */


    public Matrix(int rows, int cols , boolean random){
        data = new double[rows][cols];
        this.rows=rows;
        this.cols=cols;
        Random generator = new Random(10);

        if(random){
            for(int j=0;j<rows;j++)
            {
                for(int i=0;i<cols;i++)
                {
                    double ran = 0;
                    while(ran == 0){
//                        ran = Math.random()*2-1;
                    ran = generator.nextDouble()*2-1;
                        data[j][i]=ran;
                    }
                }
            }

        }

    }


    public static Matrix plus_matrix(Matrix a, Matrix b) {
        Matrix temp=new Matrix(a.rows,a.cols , false);
        for(int j=0;j<a.rows;j++)
        {
            for(int i=0;i<a.cols;i++)
            {
                temp.data[j][i]=a.data[j][i]+b.data[j][i];
            }
        }
        return temp;
    }
//    public static Matrix transpose(Matrix a) {
//        Matrix temp=new Matrix(a.cols,a.rows , false);
//        for(int i=0;i<a.rows;i++)
//        {
//            for(int j=0;j<a.cols;j++)
//            {
//                temp.data[j][i]=a.data[i][j];
//            }
//        }
//        return temp;
//    }
//    public static Matrix multiply(Matrix a, Matrix b) {
//        Matrix temp=new Matrix(a.rows,b.cols , false);
//        for(int i=0;i<temp.rows;i++)
//        {
//            for(int j=0;j<temp.cols;j++)
//            {
//                double sum=0;
//                for(int k=0;k<a.cols;k++)
//                {
//                    sum+=a.data[i][k]*b.data[k][j];
//                }
//                temp.data[i][j]=sum;
//            }
//        }
//        return temp;
//    }
//    public void multiply(double a) {
//        for(int i=0;i<rows;i++)
//        {
//            for(int j=0;j<cols;j++)
//            {
//                this.data[i][j]*=a;
//            }
//        }
//
//    }

    public void set(int row, int col, double value) {
      this.data[row][col] = value;
    }

//    public void add(int row, int col, double value) {
//        this.data[row][col] += value;
//    }
}
