#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <TCanvas.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TFile.h>

//======================================= Simulation Parameter Setting ===================================================//
const uint16_t N_Bit            = 12;                //The DAC's bits
const uint16_t N_Code           = 1 << N_Bit;        //The input code range 0 ~ 4095
const uint16_t N_DNL            = N_Code - 1;        //The DNL num 4095
const uint16_t N_MC             = 500;               //Number of Monte Carlo simulations
const double   Sigma_u          = 0.01;              //Mismatch of a unit current source
const double   I_Unit_Ideal     = 1.0;               //Ideal single‑unit current source current
//========================================================================================================================//

/*
 * std::vector<double> getBinaryDAC_current(std::mt19937& engine)
 * Generate an actual output current array for a fully binary DAC
 * Principle: Each bit in binary is made up of 2^k unit current sources in parallel,each unit adds Gaussian random mismatch
 * Input    : Random number engine
 * Output   : vector<double> Iout[N_Code],Iout[c]:the DAC output current corresponding to the digital code c
 */
std::vector<double> getBinaryDAC_current(std::default_random_engine& engine){
    std::normal_distribution<double> gauss(0.0,Sigma_u);
    std::vector<double> Iout(N_Code,0.0);
    std::vector<uint16_t> lenBit(N_Bit);
    for(uint16_t b = 0; b < N_Bit; b++){
        lenBit[b] = 1 << b;
    }
    const uint16_t N_Unit_Mis = N_Code - 1;
    std::vector<double> I_Unit_Mis(N_Unit_Mis);
    for(uint16_t i = 0; i < N_Unit_Mis; i++){
        I_Unit_Mis[i] = I_Unit_Ideal*(1.0 + gauss(engine));
    }
    for(uint16_t i = 0; i < N_Code; i++){
        double sum_current = 0.0;
        uint16_t ptr = 0;
        for(uint16_t b = 0; b < N_Bit; b++){
            if((i >> b) & 1){
                for(uint16_t s = 0; s < lenBit[b]; s++){
                    sum_current += I_Unit_Mis[ptr++];
                }
            }
            else {
                ptr += lenBit[b];
            }
        }
        Iout[i] = sum_current;
    }
    return Iout;
}

/*
 * std::vector<double> calDNL(const std::vector<double>& Iout,const double& LSB_Ideal)
 * Calculate DNL
 * Iout: DAC output current sequence
 * LSB_Ideal: Ideal LSB Current
 * return DNL unit LSB
 */
std::vector<double> calDNL(const std::vector<double>& Iout,const double& LSB_Ideal){
    std::vector<double> DNL(N_DNL);
    for(uint16_t i = 0; i < N_DNL; i++){
        double delta_out = Iout[i+1] - Iout[i];
        DNL[i] = (delta_out / LSB_Ideal) - 1.0;
    }
    return DNL;
}

/*
 * std::vector<double> calINL_Endpoint(const std::vector<double>& Iout,const double& LSB_Ideal)
 * Endpoint Method Calculating INL (without using the least squares method)
 * Endpoint method principle: the fitted line is forced to pass through the two endpoints (code=0, Iout[0]) and (code=4095, Iout[4095])
 * Fitted line: I_endfit(c) = G_end * c
 * Gain G_end = (Iout[N_CODE‑1] − Iout[0]) / (N_CODE‑1)
 * INL(c) = (I_actual(c) − I_endfit(c)) / LSB_ideal
 */
std::vector<double> calINL_Endpoint(const std::vector<double>& Iout,const double& LSB_Ideal){
    std::vector<double> INL(N_Code);
    double I0 = Iout[0];
    double Iend = Iout[N_Code - 1];
    double G_end = (Iend - I0)/(N_Code - 1);
    for(uint16_t i = 0; i < N_Code; i++){
        double I_fit = I0 + G_end * i;
        INL[i] = (Iout[i] - I_fit)/LSB_Ideal;
    }
    return INL;
}

int main(){
    std::default_random_engine engine(time(0));
    const double LSB_Ideal = I_Unit_Ideal;
    std::vector<std::vector<double>> DNL_all(N_MC, std::vector<double>(N_DNL));
    std::vector<std::vector<double>> INL_all(N_MC, std::vector<double>(N_Code));
    std::cout<<"========================binary DAC Monte Carlo start======================"<<std::endl;
    for(uint16_t imc = 0; imc < N_MC; imc++){
        std::vector<double> Iout = getBinaryDAC_current(engine);
        std::vector<double> DNL  = calDNL(Iout, LSB_Ideal);
        std::vector<double> INL  = calINL_Endpoint(Iout, LSB_Ideal);
        DNL_all[imc] = DNL;
        INL_all[imc] = INL;
        if((imc+1)%100 == 0){
            std::cout << "MC run : " << imc+1 << " / " << N_MC << std::endl;
        }
    }
    std::vector<double> DNL_RMS_Code(N_DNL,0.0);
    std::vector<double> INL_RMS_Code(N_Code,0.0);
    for(int c = 0; c < N_DNL; c++) {
        double sum2 = 0.0;
        for(int m = 0; m < N_MC; m++)
        {
          sum2 += DNL_all[m][c] * DNL_all[m][c];
        }
        DNL_RMS_Code[c] = std::sqrt( sum2 / N_MC );
    }
    for(int c = 0; c < N_Code; c++){
        double sum2 = 0.0;
        for(int m = 0; m < N_MC; m++)
        {

          sum2 += INL_all[m][c] * INL_all[m][c];
        }
        INL_RMS_Code[c] = std::sqrt( sum2 / N_MC );
    }

    // Open ROOT output file for storing histograms and graphs
    TFile* root_outfile = new TFile("BinaryDAC_Endpoint.root", "RECREATE");

    //===== 2×2 canvas for four sub‑plots =====
    TCanvas* canvas4 = new TCanvas("canvas4","Binary DAC 4‑plot",1400,1000);
    canvas4->Divide(2,2);

    //----------Subplot 1: Overlay raw DNL curves of all 500 Monte‑Carlo runs------------
    canvas4->cd(1);
    gPad->SetGrid();
    for(uint16_t imc=0; imc<N_MC; imc++)
    {
        TGraph* g = new TGraph(N_DNL);
        for(int ip=0;ip<N_DNL;ip++){
            g->SetPoint(ip, ip, DNL_all[imc][ip]);
        }
        g->SetLineColor(kBlack);
        g->SetLineWidth(1);
        if(imc==0){
            g->SetTitle("DNL all MC raw;DAC Code;DNL [LSB]");
            g->SetMinimum(-2.0);
            g->SetMaximum(2.0);
            g->Draw("AL");
        }else{
            g->Draw("L");
        }
    }

    //----------Subplot 2: Overlay raw INL curves of all 500 Monte‑Carlo runs------------
    canvas4->cd(2);
    gPad->SetGrid();
    for(uint16_t imc=0; imc<N_MC; imc++)
    {
        TGraph* g = new TGraph(N_Code);
        for(int ip=0;ip<N_Code;ip++){
            g->SetPoint(ip, ip, INL_all[imc][ip]);
        }
        g->SetLineColor(kBlack);
        g->SetLineWidth(1);
        if(imc==0){
            g->SetTitle("INL all MC raw;DAC Code;INL [LSB]");
            g->SetMinimum(-2.0);
            g->SetMaximum(2.0);
            g->Draw("AL");
        }else{
            g->Draw("L");
        }
    }

    //----------Subplot 3: Per‑code RMS of DNL------------
    canvas4->cd(3);
    gPad->SetGrid();
    TGraph* gr_dnl_rms = new TGraph(N_DNL);
    for(int i=0; i<N_DNL; i++){
        gr_dnl_rms->SetPoint(i, i, DNL_RMS_Code[i]);
    }
    gr_dnl_rms->SetTitle("DNL per‑code RMS;DAC Code;DNL RMS [LSB]");
    gr_dnl_rms->SetLineColor(kRed);
    gr_dnl_rms->SetLineWidth(2);
    gr_dnl_rms->Draw("AL");

    //----------Subplot 4: Per‑code RMS of endpoint‑based INL------------
    canvas4->cd(4);
    gPad->SetGrid();
    TGraph* gr_inl_rms = new TGraph(N_Code);
    for(int i=0; i<N_Code; i++){
        gr_inl_rms->SetPoint(i, i, INL_RMS_Code[i]);
    }
    gr_inl_rms->SetTitle("Endpoint INL per‑code RMS;DAC Code;INL RMS [LSB]");
    gr_inl_rms->SetLineColor(kRed);
    gr_inl_rms->SetLineWidth(2);
    gr_inl_rms->Draw("AL");


    //======= Export figure to disk (WSL cannot pop‑up native GUI window) =======
    canvas4->SaveAs("BinaryDAC_4plot.png");

    // Write canvas and graphs into root file
    canvas4->Write();
    gr_dnl_rms->Write();
    gr_inl_rms->Write();

    root_outfile->Close();

    std::cout << "\nAll picture & root file saved.\n";
    std::cout << "Output file: BinaryDAC_4plot.png , BinaryDAC_Endpoint.root\n";

    return 0;
}
