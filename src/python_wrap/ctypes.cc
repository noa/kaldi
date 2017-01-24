#include <iostream>

#include "util/table-types.h"
#include "hmm/posterior.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {
    typedef SequentialBaseFloatMatrixReader SBFMReader;
    typedef Matrix<BaseFloat> MatrixF;
    typedef RandomAccessPosteriorReader RAPReader;
}

extern "C" {
    using namespace kaldi;

    /****************************** SBFMReader ******************************/

    //SequentialTableReader(): impl_(NULL) { }
    SBFMReader* SBFMReader_new() {
        return new SBFMReader();
    }
    //SequentialTableReader(const std::string &rspecifier);
    SBFMReader* SBFMReader_new_char(char * rspecifier) {
        return new SBFMReader(rspecifier);
    }
    //bool Open(const std::string &rspecifier);
    int SBFMReader_Open(SBFMReader* r, char * rspecifier) {
        return r->Open(rspecifier);
    }
    //inline bool Done();
    int SBFMReader_Done(SBFMReader* r) {
        return r->Done();
  }
    //inline std::string Key();
    const char * SBFMReader_Key(SBFMReader* r) {
        return r->Key().c_str();
    }
    //void FreeCurrent();
    void SBFMReader_FreeCurrent(SBFMReader* r) {
        r->FreeCurrent();
    }
    //const T &Value();
    const MatrixF * SBFMReader_Value(SBFMReader* r) {
        return &r->Value(); //despite how dangerous this looks, this is safe because holder maintains object (it's not stack allocated)
    }
    //void Next();
    void SBFMReader_Next(SBFMReader* r) {
        r->Next();
    }
    //bool IsOpen() const;
    int SBFMReader_IsOpen(SBFMReader* r) {
        return r->IsOpen();
    }
    //bool Close();
    int SBFMReader_Close(SBFMReader* r) {
        return r->Close();
    }
    //~SequentialTableReader();
    void SBFMReader_Delete(SBFMReader* r) {
        delete r;
    }

    /****************************** MatrixF ******************************/

    //NumRows ()
    int MatrixF_NumRows(MatrixF *m) {
        return m->NumRows();
    }
    //NumCols ()
    int MatrixF_NumCols(MatrixF *m) {
        return m->NumCols();
    }

    //Stride ()
    int MatrixF_Stride(MatrixF *m) {
        return m->Stride();
    }

    void MatrixF_cpy_to_ptr(MatrixF *m, float * dst, int dst_stride) {
        int num_rows = m->NumRows();
        int num_cols = m->NumCols();
        int src_stride = m->Stride();
        int bytes_per_row = num_cols * sizeof(float);

        float * src = m->Data();

        for (int r=0; r<num_rows; r++) {
            memcpy(dst, src, bytes_per_row);
            src += src_stride;
            dst += dst_stride;
        }
    }

    //SizeInBytes ()
    int MatrixF_SizeInBytes(MatrixF *m) {
        return m->SizeInBytes();
    }
    //Data (), Real is usually float32
    const float * MatrixF_Data(MatrixF *m) {
        return m->Data();
    }

    /****************************** RAPReader ******************************/

    RAPReader* RAPReader_new_char(char * rspecifier) {
        return new RAPReader(rspecifier);
    }

    //bool  HasKey (const std::string &key)
    int RAPReader_HasKey(RAPReader* r, char * key) {
        return r->HasKey(key);
    }

    //const T &   Value (const std::string &key)
    int * RAPReader_Value(RAPReader* r, char * key) {
        //return &r->Value(key);
        const Posterior p = r->Value(key);
        int num_rows = p.size();
        if (num_rows == 0) {
            return NULL;
        }

        //std::cout << "num_rows " << num_rows << std::endl;

        int * vals = new int[num_rows];

        for (int row=0; row<num_rows; row++) {
            int num_cols = p.at(row).size();
            if (num_cols != 1) {
                std::cout << "num_cols != 1: " << num_cols << std::endl;
                delete vals;
                return NULL;
            }
            std::pair<int32, BaseFloat> pair = p.at(row).at(0);
            if (pair.second != 1) {
                std::cout << "pair.second != 1: " << pair.second << std::endl;
                delete vals;
                return NULL;
            }
            vals[row] = pair.first;
        }

        return vals;
    }

    void RAPReader_DeleteValue(RAPReader* r, int * vals) {
        delete vals;
    }

    //~RandomAccessTableReader ()
    void RAPReader_Delete(RAPReader* r) {
        delete r;
    }
}
