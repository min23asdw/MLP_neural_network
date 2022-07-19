

import java.io.*;
import java.util.ArrayList;

public class main {

    public static void main(String[] args) throws IOException {
        ArrayList<Double[]> test_dataset_1 = new ArrayList<>();
        ArrayList<Double[]> test_desired_data_1 = new ArrayList<>();

        ArrayList<Double[]> train_dataset_1 = new ArrayList<>();
        ArrayList<Double[]> train_desired_data_1 = new ArrayList<>();

        FileInputStream fstream = new FileInputStream("src/Flood_dataset.txt");
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        String data;

        double maxInput = 0;
        double minInput =  100000;
        double maxdesired = 0;
        double mindesired =  100000;

        for(int tain_i = 0 ; tain_i < 1 ; tain_i ++) {

            int line_i = 0;
            while ((data = br.readLine()) != null) { // each line
                String[] tmp = data.split("\t");    //Split space

                Double[] tmp_test_dataset = new Double[tmp.length-1];
                Double[] tmp_test_desired_data = new Double[1];

                Double[] tmp_train_dataset = new Double[tmp.length-1];
                Double[] tmp_train_desired_data = new Double[1];

                int word_i = 0;
                for (String t : tmp) {  // each word
                    double tmp_val = Double.parseDouble(t);

                    if (line_i % 10 == tain_i) { // test 10% data
                        if (word_i == tmp.length - 1) { //  desired_data
                            tmp_test_desired_data[0] = tmp_val;
                        } else {
                            tmp_test_dataset[word_i] = tmp_val;
                        }

                    } else { //train 90%
                        if (word_i == tmp.length - 1) { //train_dataset
                            tmp_train_desired_data[0] = tmp_val;
                        } else {  // train_dataset
                            tmp_train_dataset[word_i] = tmp_val;
                        }
                    }
                    word_i++;
                }

                if(tmp_test_dataset[0]!=null) test_dataset_1.add(tmp_test_dataset);
                if(tmp_test_desired_data[0]!=null) test_desired_data_1.add(tmp_test_desired_data);

                if(tmp_train_dataset[0]!= null) train_dataset_1.add(tmp_train_dataset);
                if(tmp_train_desired_data[0]!=null) train_desired_data_1.add(tmp_train_desired_data);

                line_i++;
            }
        }


        brain b1 = new brain("521", train_dataset_1 ,train_desired_data_1 ,2000,0.000001);
        b1.train();

    }
}
