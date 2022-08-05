#include "reefclient.h"

REEFDISBClient::REEFDISBClient(const Json::Value &config) {
    model_name = config["model_name"].asString();
    priority = config["priority"].asInt();
}

REEFDISBClient::~REEFDISBClient() {

}

void REEFDISBClient::init() {
    client.reset(new client::REEFClient(DEFAULT_REEF_ADDR));
    assert(client.get() != nullptr);
    client->init(priority == 0);
    model = client->load_model(
        std::string(REEF_RESOURCE_DIR) + model_name, 
        model_name
    );
    assert(model.get() != nullptr);
    auto input_blob = model->get_input_blob();
    auto output_blob = model->get_output_blob();
}

void REEFDISBClient::copyInput() {
    model->get_output();
}

void REEFDISBClient::infer() {
    assert(model.get() != nullptr);
    model->infer();
}

void REEFDISBClient::copyOutput() {
    model->load_input();
}

std::shared_ptr<DISB::Client> reef_client_factory(const Json::Value &config) {
    auto client = std::make_shared<REEFDISBClient>(config);
    return client;
}
