#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>

#define CONST_E 2.7182818284590452353602874713527

//network with nodes 784-32-16-10
//  and with weights 25088-512-160
struct node{
    float value = 0;
    float bias = 0;
};

struct images{
    float pixels[784];
    unsigned char label;
};



void initialize_array(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    neurons[0].resize(784);
    neurons[1].resize(32);
    neurons[2].resize(16);
    neurons[3].resize(10);
    for(int j = 0; j < 4; j++)
        for(auto & i : neurons[j])
            i.bias = rand() % 10 - 5;

    weights[0].resize(25088);
    weights[1].resize(512);
    weights[2].resize(160);
    for(int j = 0; j < 3; j++)
        for(float & i : weights[j])
            i = rand() % 10 - 5;
}

void loadImages(const std::string& image_path, const std::string& label_path, std::vector<images>& all_images){
    std::ifstream images_file, label_file;
    char* images_data;
    char* label_data;

    images_file.open(image_path, std::ios::in);
    label_file.open(label_path, std::ios::in);
    if(!images_file.is_open() || !label_file.is_open())
        throw;//don't have time to throw a meaningful exception so i just throw lol

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

    size = ((unsigned char)images_data[6] << 8) + (unsigned char)images_data[7];


    for(int i = 0; i < size; i++){
        all_images.resize(size);
        all_images[i].label = label_data[i + 8];
        for(int j = 0; j < 784; j++){
            all_images[i].pixels[j] = (float)images_data[i * 784 + j + 16] / 255;
        }
    }
    delete []images_data;
    delete []label_data;
}

float sigmoid_func(float input){
    return 1 / (1 + pow(CONST_E, -input));
}


void guess_image(images& image, std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights){
    for(int i = 0; i < 784; i++){
        neurons[0][i].value = image.pixels[i];
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < neurons[i + 0].size() * neurons[i + 1].size(); j++){
            neurons[i + 1][j / neurons[i].size()].value += neurons[i][j %  neurons[i].size()].value * weights[i][j];
            if((j + 1) %  neurons[i].size() == 0){
                neurons[i + 1][j / neurons[i].size()].value += neurons[i + 1][j / neurons[i].size()].bias;
                neurons[i + 1][j / neurons[i].size()].value = sigmoid_func(neurons[i + 1][j / neurons[i].size()].value);
            }
        }
    }

}

void loadWeights_biases(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, const std::string& path){
    char* data = new char[106408];

    std::ifstream wbfile;
    wbfile.open(path, std::ios::in);
    wbfile.read(data, 106408);
    wbfile.close();


    int index = 0;
    for(auto & neuron_layer : neurons){
        for(auto & neuron : neuron_layer){
            neuron.bias = *(float*)&data[index];
            index += 4;
        }
    }
    for(auto & weight_layer : weights){
        for(auto & weight : weight_layer){
            weight = *(float*)&data[index];
            index += 4;
        }
    }

    delete[] data;
}

void saveWeights_biases(std::vector<std::vector<node>>& neurons, std::vector<std::vector<float>>& weights, const std::string& path){

    char* data = new char[106408];
    int index = 0;
    for(auto & neuron_layer : neurons){
        for(auto & neuron : neuron_layer){
            *(float*)&data[index] = neuron.bias;
            index += 4;
        }
    }
    for(auto & weight_layer : weights){
        for(auto & weight : weight_layer){
            *(float*)&data[index] = weight;
            index += 4;
        }
    }

    std::ofstream wbfile;
    wbfile.open(path, std::ios::out | std::ios::trunc);
    wbfile.write(data, 106408);
    wbfile.close();

    delete[] data;
}


int main(int argc, char *argv[]) {
    srand(time(0));
    int image_to_guess = 13;
    std::vector<std::vector<node>> neurons(4);
    std::vector<std::vector<float>> weights(3);
    std::vector<images> all_images;
    std::string training_path = "/home/nejc/CLionProjects/Neural_network/AI_images_set";

    initialize_array(neurons, weights);
    loadImages(training_path + "/training/train-images.data", training_path + "/training/train-labels.data", all_images);
    if(argc != 3)
        throw;//don't have time to throw a meaningful exception so i just throw lol
    else{
        if(strcmp(argv[1], "true") == 0){
            loadWeights_biases(neurons, weights, training_path + "/neural_data.wbdata");
        }
        if(strcmp(argv[2], "true") == 0){
            saveWeights_biases(neurons, weights, training_path + "/neural_data.wbdata");
        }
    }


    guess_image(all_images[image_to_guess], neurons, weights);

    for(int i = 0; i < 10; i++){
        std::cout << neurons[3][i].value << " ";
    }
    int best_guess_number;
    float best_guess_value = 0;

    for(int i = 0; i < 10; i++){
        if(neurons[3][i].value > best_guess_value){
            best_guess_value = neurons[3][i].value;
            best_guess_number = i;
        }
    }
    std::cout << "\n" << best_guess_number << "\n" << (int)all_images[image_to_guess].label;



    return 0;
}
