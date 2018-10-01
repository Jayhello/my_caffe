//
// Created by root on 9/12/18.
//

#ifndef MY_CAFFE_LAYER_FACTORY_H
#define MY_CAFFE_LAYER_FACTORY_H

#include <map>
#include <string>
#include "common.h"
#include "layer.h"
#include "caffe.pb.h"

namespace caffe{

template<typename Dtype>
class LayerRegistry{
public:
    // 函数指针Creator，返回的是Layer<Dtype>类型的指针
    typedef shared_ptr<Layer<Dtype>> (*Creator)(const LayerParameter&);

    // CreatorRegistry是字符串与对应的Creator的映射
    typedef std::map<string, Creator> CreatorRegistry;

    static CreatorRegistry& Registry() {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }

    // Adds a creator.
    // 给定类型，以及函数指针，加入到注册表
    static void AddCreator(const string& type, Creator creator) {
        CreatorRegistry& registry = Registry();

//        CHECK_EQ(registry.count(type), 0)
//            << "Layer type " << type << " already registered.";

        LOG(INFO) << "AddCreator: " << type;
        registry[type] = creator;
    }

    // Get a layer using a LayerParameter.
    // 通过LayerParameter，返回特定层的实例智能指针
    static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
        if (Caffe::root_solver()) {//TODO don't know why?
            LOG(INFO) << "Creating layer " << param.name();
        }

        const string& type = param.type();
        CreatorRegistry& registry = Registry();

        CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
                                          << " (known types: " << LayerTypeListString() << ")";
        return registry[type](param);
    }

    // get all the registered layer in a vector<string>
    static vector<string> LayerTypeList() {
        CreatorRegistry& registry = Registry();
        vector<string> layer_types;
        for (typename CreatorRegistry::iterator iter = registry.begin();
             iter != registry.end(); ++iter) {
            layer_types.push_back(iter->first);
        }
        return layer_types;
    }

private:
    
    // convert all the registered layer name to a single string
    static string LayerTypeListString() {
        vector<string> layer_types = LayerTypeList();
        string layer_types_str;
        for (vector<string>::iterator iter = layer_types.begin();
             iter != layer_types.end(); ++iter) {
            if (iter != layer_types.begin()) {
                layer_types_str += ", ";
            }
            layer_types_str += *iter;
        }
        return layer_types_str;
    }

    // Layer registry should never be instantiated - everything is done with its
    // static variables.
    // 私有构造函数，禁止实例化，所有功能都由静态函数完成，所以不需要实例化
    LayerRegistry() {}
    
};

    template <typename Dtype>
    class LayerRegisterer {
    public:
        LayerRegisterer(const string& type,
                        shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&))
        {
            // LOG(INFO) << "Registering layer type: " << type;
            LayerRegistry<Dtype>::AddCreator(type, creator);
        }
    };

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

/*
 * 宏 REGISTER_LAYER_CLASS 为每个type生成了create方法，并和type一起注册到了LayerRegistry中
 * ，保存在一个map里面。
 */
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
    
}


#endif //MY_CAFFE_LAYER_FACTORY_H
