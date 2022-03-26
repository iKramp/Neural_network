#include <iostream>
#include <vector>
#include <string>
#include <fstream>

//network with nodes 784-32-16-10
//  and with weights 25088-512-160
struct node{
    float value;
    float bias;
};

struct images{
    float pixels[784];
    unsigned char label;
};

void initialize_array(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    neurons[0].resize(784);
    for(int i = 0; i < 784; i++){
        neurons[0][i].bias = 0;
    }
    neurons[1].resize(32);
    for(int i = 0; i < 32; i++){
        neurons[1][i].bias = 0;
    }
    neurons[2].resize(16);
    for(int i = 0; i < 16; i++){
        neurons[2][i].bias = 0;
    }
    neurons[3].resize(10);
    for(int i = 0; i < 10; i++){
        neurons[3][i].bias = 0;
    }

    weights[0].resize(25088);
    for(int i = 0; i < 25088; i++){
        weights[0][i] = 0;
    }
    weights[1].resize(512);
    for(int i = 0; i < 512; i++){
        weights[1][i] = 0;
    }
    weights[2].resize(160);
    for(int i = 0; i < 160; i++){
        weights[2][i] = 0;
    }
}

void loadImages(std::string image_path, std::string label_path, std::vector<images> all_images){
    std::ifstream images_file, label_file;
    char* images_data;
    char* label_data;

    images_file.open(image_path, std::ios::in);
    label_file.open(label_path, std::ios::in);
    if(!images_file.is_open() || !label_file.is_open())
        throw;

    images_file.seekg(0, std::ios::end);
    long size = (int)images_file.tellg();
    images_data = new char[size];
    images_file.seekg(0, std::ios::beg);
    images_file.read(images_data, size);

    label_file.seekg(0, std::ios::end);
    size = (int)label_file.tellg();
    label_data = new char[size];
    label_file.seekg(0, std::ios::beg);
    label_file.read(label_data, size);

    images_file.close();
    label_file.close();

    //size = (images_data[4] << 24) + (images_data[5] << 16) + (images_data[6] << 8) + (images_data[7]);
    size = 0;
    size += (unsigned char)images_data[6] << 8;
    size += (unsigned char)images_data[7];


    for(int i = 0; i < size; i++){
        all_images.resize(size);
        all_images[i].label = label_data[i + 8];
        unsigned char test = label_data[i + 8];
        for(int j = 0; j < 784; j++){
            int offset = i * 784 + j + 16;//168
            float pixel = (float)images_data[i * 784 + j + 16] / 255;
            all_images[i].pixels[j] = (float)images_data[i * 784 + j + 16] / 255;
        }
    }
}


int main() {
    std::vector<std::vector<node>> neurons(4);
    std::vector<std::vector<float>> weights(3);
    std::vector<images> all_images;
    initialize_array(neurons, weights);
    std::string training_path = "/home/nejc/CLionProjects/Neural_network/AI_images_set/training";
    loadImages(training_path + "/train-images.data", training_path + "/train-labels.data", all_images);




    return 0;
}
