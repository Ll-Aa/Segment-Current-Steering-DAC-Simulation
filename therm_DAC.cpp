#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <cstdint>
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1D.h"
#include "TFile.h"

//===========================================Simulation Parameter Setting==============================//
const uint16_t N_Bit          = 12;           //The DAC's bits
const uint16_t N_Code         = 1 << N_Bit;   //The input code range 0 ~ 4095
const uint16_t N_DNL          = N_Code - 1;   //The DNL num 4095
const uint16_t N_MC           = 500;          //Number of Monte Carlo simulations
const double   Sigma_u        = 0.01;         //Mismatch of a unit current source
const double   I_Unit_Ideal   = 1.0;           //Ideal single unit current source current
//====================================================================================================//

/*
 * std::vector<double> getThermDAC_current(std::mt19937& engine)
 * Generate an actual output current array for a fully thermometer DAC
 * Input :Random engine
 * Output:vector<double> Iout[N_Code],Iout[c]:the DAC output current corresponding to the digital code c
 */
std::vector<double> getThermDAC_current(std::default_random_engine& engine){
  std::normal_distribution<double> gauss(0.0,Sigma_u);
  std::vector<double> Iout(N_Code,0.0);
  std::vector<double> I_Unit_Mis(N_Code);
  const uint16_t N_Unit_Mis = N_Code -1;
  double sum = 0.0;
  for(uint16_t c = 0; c < N_Unit_Mis; c++){
    I_Unit_Mis[c] = I_Unit_Ideal*(1.0 + gauss(engine));
  }
  Iout[0] = 0.0;
  for(uint16_t i = 1; i < N_Code; i++){
    sum += I_Unit_Mis[i-1];
    Iout[i] = sum;
  }
  return Iout;
}

/*
 * std::vector<double> calDNL(const std::vector<double>& Iout,const double& LSB_Ideal)
 * Calculate DNL
 * Iout: DAC output current sequence
 * LSB_Ideal:Ideal LSB Current
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
 * Fitted line : I_endfit(c) = G_end*c
 * Gain G_end = (Iout[N_Code - 1] - Iout[0]) / (N_Code - 1)
 * INL(c) = (I_actual(c) - I_endfit(c)) / LSB_Ideal
 */
std::vector<double> calINL_Endpoint(const std::vector<double>& Iout,const double& LSB_Ideal){
  std::vector<double> INL(N_Code);
  double I0 = Iout[0];
  double Iend = Iout[N_Code - 1];
  double G_end = (Iend - I0) / (N_Code - 1);
  for(uint16_t i = 0;i < N_Code; i++){
    double I_fit = I0 + G_end*i;
    INL[i] = (Iout[i] - I_fit) / LSB_Ideal;
  }
  return INL;
}

int main(){
  std::default_random_engine engine(time(0));
  const double LSB_Ideal = I_Unit_Ideal;
  std::vector<std::vector<double>> DNL_all(N_MC,std::vector<double>(N_DNL));
  std::vector<std::vector<double>> INL_all(N_MC,std::vector<double>(N_Code));
  std::cout<<"===========================Therm DAC Monte Carlo start=============================="<<std::endl;
  for(uint16_t imc = 0; imc < N_MC; imc++){
    std::vector<double> Iout = getThermDAC_current(engine);
    std::vector<double> DNL = calDNL(Iout,LSB_Ideal);
    std::vector<double> INL = calINL_Endpoint(Iout,LSB_Ideal);
    DNL_all[imc] = DNL;
    INL_all[imc] = INL;
  }
  std::vector<double> DNL_RMS_Code(N_DNL,0.0);
  std::vector<double> INL_RMS_Code(N_Code,0.0);
  for(int c = 0; c < N_DNL; c++){
    double sum = 0.0;
    for(int m = 0; m < N_MC; m++){
      sum += DNL_all[m][c] * DNL_all[m][c];
    }
    DNL_RMS_Code[c] = std::sqrt(sum / N_MC);
  }
  for(int c = 0; c < N_Code; c++){
    double sum = 0.0;
    for(int m = 0; m < N_MC; m++){
      sum += INL_all[m][c] * INL_all[m][c];
    }
    INL_RMS_Code[c] = std::sqrt(sum / N_MC);
  }
  TFile* root_outfile = new TFile("ThermometerDAC_Endpoint.root","RECREATE");
  TCanvas* canvas4 = new TCanvas("cancas4","Therm DAC 4-plot",1400,1000);
  canvas4->Divide(2,2);

  canvas4->cd(1);
  gPad->SetGrid();
  for(uint16_t imc = 0; imc < N_MC; imc++){
    TGraph* g = new TGraph(N_DNL);
    for(int ip = 0; ip < N_DNL; ip++){
      g->SetPoint(ip,ip,DNL_all[imc][ip]);
    }
    g->SetLineColor(kBlack);
    g->SetLineWidth(1);
    if(imc == 0){
      g->SetTitle("DNL all MC raw;DAC Code;DNL [LSB]");
      g->SetMinimum(-2.0);
      g->SetMaximum(2.0);
      g->Draw("AL");
    }
    else {
      g->Draw("L");
    }
  }

  canvas4->cd(2);
  gPad->SetGrid();
  for(uint16_t imc = 0; imc < N_MC; imc++){
    TGraph* g = new TGraph(N_Code);
    for(int ip = 0; ip < N_Code; ip++){
      g->SetPoint(ip,ip,INL_all[imc][ip]);
    }
    g->SetLineColor(kBlack);
    g->SetLineWidth(1);
    if(imc == 0){
      g->SetTitle("INL all MC raw;DAC Code;INL [LSB]");
      g->SetMinimum(-2.0);
      g->SetMaximum(2.0);
      g->Draw("AL");
    }
    else {
      g->Draw("L");
    }
  }

  canvas4->cd(3);
  gPad->SetGrid();
  TGraph* gr_dnl_rms = new TGraph(N_DNL);
  for(int i = 0; i < N_DNL; i++){
    gr_dnl_rms->SetPoint(i,i,DNL_RMS_Code[i]);
  }
  gr_dnl_rms->SetTitle("DNL per-code RMS;DAC Code;DNL RMS [LSB]");
  gr_dnl_rms->SetLineColor(kRed);
  gr_dnl_rms->SetLineWidth(2);
  gr_dnl_rms->Draw("AL");

  canvas4->cd(4);
  gPad->SetGrid();
  TGraph* gr_inl_rms = new TGraph(N_Code);
  for(int i = 0; i < N_Code; i++){
    gr_inl_rms->SetPoint(i,i,INL_RMS_Code[i]);
  }
  gr_inl_rms->SetTitle("INL per-code RMS;DAC Code;INL RMS [LSB]");
  gr_inl_rms->SetLineColor(kRed);
  gr_inl_rms->SetLineWidth(2);
  gr_inl_rms->Draw("AL");

  canvas4->SaveAs("ThermDAC_4plot.png");
  canvas4->Write();
  gr_dnl_rms->Write();
  gr_inl_rms->Write();
  root_outfile->Close();
  
  return 0;
}
