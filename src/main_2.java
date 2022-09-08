import java.io.*;
import java.util.ArrayList;

public class main_2 {

    private static ArrayList<ArrayList<Double[]>> test_dataset = new ArrayList<>();
    private static ArrayList<ArrayList<Double[]>> test_desired_data = new ArrayList<>();

    private static ArrayList<ArrayList<Double[]>> train_dataset = new ArrayList<>();
    private static ArrayList<ArrayList<Double[]>> train_desired_data = new ArrayList<>();

    private static int NumberOftest = 10;

    public static void main(String[] args) throws IOException {
        for(int tain_i = 0 ; tain_i < NumberOftest ; tain_i ++) {
            ArrayList<Double[]> test_dataset_i = new ArrayList<>();
            ArrayList<Double[]> test_desired_data_i = new ArrayList<>();

            ArrayList<Double[]> train_dataset_i = new ArrayList<>();
            ArrayList<Double[]> train_desired_data_i = new ArrayList<>();

            FileInputStream fstream = new FileInputStream("src/cross.pat");
            DataInputStream in = new DataInputStream(fstream);
            BufferedReader br = new BufferedReader(new InputStreamReader(in));
            String data;

            int line_i = 1;
            while ((data = br.readLine()) != null) { // each line
                if(line_i%3 == 0 || (line_i+1)%3 == 0) { // not p line
                String[] eachLine = data.split("\\s+");
                Double[] temp = new Double[eachLine.length];

                for(int  i =0 ; i<eachLine.length;i++){
                    double dataNum = Double.parseDouble(eachLine[i]);
                    temp[i] = dataNum ;
                }

                if (line_i % 3 == 0) {  // line desired
                    if (line_i % 10 == tain_i) {  // 10% for test
                        test_desired_data_i.add(temp);
                    } else
                        train_desired_data_i.add(temp);
                } else if ((line_i + 1) % 3 == 0) { // line input
                    if ((line_i + 1) % 10 == tain_i) { // 10% for test
                        test_dataset_i.add(temp);
                    } else
                        train_dataset_i.add(temp);
                }
            }
                line_i++;
            }
            test_desired_data.add(test_desired_data_i);
            train_desired_data.add(train_desired_data_i);
            test_dataset.add(test_dataset_i);
            train_dataset.add(train_dataset_i);
        }


        for(int test_i = 0 ; test_i < NumberOftest ; test_i ++) {
            System.out.println("===================================================");
            brain_2 b2 = new brain_2("2,8,2", 3000, 0.00007, 1, 0.01, 0.1);
            System.out.println("train: " + test_i);
            b2.train( train_dataset.get(test_i), train_desired_data.get(test_i));
            System.out.println("test: " + test_i);
            b2.test(test_dataset.get(test_i), test_desired_data.get(test_i));
            System.out.println("===================================================");
        }
    }
}
