#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <tuple>
#include <Eigen/Dense>
#include <TCanvas.h>
#include <TGraph.h>
#include <TFile.h>
#include <TLegend.h>
#include <TAxis.h>

//======================== Simulation Parameter Setting ======================//
const int    Nbit         = 12;
const int    FIX_BIN_LOW  = 3;          
const int    THERM_TOTAL  = 9;          
const int    N_code       = 1 << Nbit;
const int    N_dnl        = N_code - 1;
const int    N_mc         = 500;
const double sigma_u      = 0.01;
const double I_Unit_Ideal = 1.0;
const double LSB_Ideal    = 1.0;
const int    N_unit_total = 4095;
const double S_unit       = 0.18;
const double delta_thresh = 0.3;
//=============================================================================//
struct ResRow
{
    int m;                
    int therm_high;
    int therm_low;
    double dnl_val;
    double A1;
    double X;
    double inl_val;
    double A2;
    double Y05;
    double A2_INL1;
    double Y10;
    double A_digital;
    double A_chip_INL1;
    double log2_Achip_INL1;
};

Eigen::VectorXd calDNL(const Eigen::VectorXd& Iout)
{
    Eigen::VectorXd dnl(N_dnl);
    for(int i = 0; i < N_dnl; i++)
    {
        dnl(i) = Iout(i+1) - Iout(i) - 1.0;
    }
    return dnl;
}

Eigen::VectorXd calINL_endpoint(const Eigen::VectorXd& Iout)
{
    Eigen::VectorXd c_vec(N_code);
    for(int i = 0; i < N_code; i++) c_vec(i) = double(i);
    double G = (Iout(N_code-1) - Iout(0))/ (N_code - 1);
    Eigen::VectorXd Iout_fit(N_code);
    for(int i = 0; i < N_code; i++){
      Iout_fit(i) = Iout(0) + G*i;
    }
    Eigen::VectorXd inl = (Iout - Iout_fit) / LSB_Ideal;
    return inl;
}

Eigen::VectorXd rmsAlongMC(const Eigen::MatrixXd& mat)
{
    int nPt = mat.cols();
    Eigen::VectorXd res(nPt);
    for(int ip = 0; ip < nPt; ip++)
    {
        double sum2 = mat.col(ip).array().square().sum();
        res(ip) = std::sqrt( sum2 / double(mat.rows()) );
    }
    return res;
}

std::tuple<double,double> simulateOnePaperDAC(int m, std::default_random_engine& rng)
{
    int therm_h = m;
    int therm_l = THERM_TOTAL - m;
    int L_fixed = FIX_BIN_LOW;
    int weight_therm_low  = 1 << L_fixed;
    
    int weight_therm_high = 1 << (therm_l + L_fixed);
    

    int num_step_h = (1 << therm_h) - 1;
    int num_step_l = (1 << therm_l) - 1;
    int num_step_bin = (1 << L_fixed) - 1;

    int total_units = num_step_h * weight_therm_high + num_step_l * weight_therm_low + num_step_bin;
    if(total_units != N_unit_total)
    {
        std::cerr<<"[WARN] m="<<m<<" unit mismatch total="<<total_units<<" expect 4095\n";
    }

    Eigen::MatrixXd dnl_all(N_mc, N_dnl);
    Eigen::MatrixXd inl_all(N_mc, N_code);
    std::normal_distribution<double> gauss(1.0, sigma_u);

    for(int mc=0; mc<N_mc; mc++)
    {
        Eigen::VectorXd I_unit(N_unit_total);
        for(int iu=0; iu<N_unit_total; iu++)
        {
            I_unit(iu) = gauss(rng);
        }
        int ptr = 0;

      
        Eigen::VectorXd I_h(num_step_h);
        for(int i=0;i<num_step_h;i++)
        {
            double s=0;
            for(int k=0;k<weight_therm_high;k++) s += I_unit(ptr++);
            I_h(i)=s;
        }
        
        Eigen::VectorXd I_l(num_step_l);
        for(int i=0;i<num_step_l;i++)
        {
            double s=0;
            for(int k=0;k<weight_therm_low;k++) s += I_unit(ptr++);
            I_l(i)=s;
        }
        
        Eigen::VectorXd I_bin(L_fixed);
        for(int b=0;b<L_fixed;b++)
        {
            int w = 1 << (L_fixed-1-b);
            double s=0;
            for(int k=0;k<w;k++) s += I_unit(ptr++);
            I_bin(b)=s;
        }

        Eigen::VectorXd Iout(N_code);
        for(int code=0;code<N_code;code++)
        {
            int bin_part  = code & ((1<<L_fixed)-1);
            int therm_all = code >> L_fixed;

            int val_l = therm_all & ((1<<therm_l)-1);
            int val_h = therm_all >> therm_l;

            double Itot = 0.0;
            if(val_h>0) Itot += I_h.head(val_h).sum();
            if(val_l>0) Itot += I_l.head(val_l).sum();
            for(int b=0;b<L_fixed;b++)
            {
                if( (bin_part >> (L_fixed-1-b)) &1 )
                {
                    Itot += I_bin(b);
                }
            }
            Iout(code)=Itot;
        }
        auto dnl = calDNL(Iout);
        auto inl = calINL_endpoint(Iout);
        dnl_all.row(mc)=dnl;
        inl_all.row(mc)=inl;
    }

    auto dnl_rms = rmsAlongMC(dnl_all);
    auto inl_rms = rmsAlongMC(inl_all);
    double dnl_max = dnl_rms.maxCoeff();
    double inl_max = inl_rms.maxCoeff();
    return {dnl_max,inl_max};
}


std::vector<std::tuple<int,double,double>> simulateAllPaperDAC(std::default_random_engine& rng)
{
    std::vector<std::tuple<int,double,double>> res;
    for(int m=0;m<=THERM_TOTAL;m++)
    {
        auto [d,i] = simulateOnePaperDAC(m, rng);
        res.emplace_back(m,d,i);
    }
    return res;
}

std::vector<ResRow> computeAreaTablePaper(const std::vector<std::tuple<int,double,double>>& raw)
{
    
    int idx_ref=0;
    for(int i=0;i<(int)raw.size();i++)
    {
        if(std::get<0>(raw[i])==THERM_TOTAL){ idx_ref=i; break; }
    }
    double dnl_ref = std::get<1>(raw[idx_ref]);
    double inl_ref = std::get<1>(raw[idx_ref]);

    std::vector<ResRow> table;
    for(auto &t : raw)
    {
        int m      = std::get<0>(t);
        int th     = m;
        int tl     = THERM_TOTAL - m;
        double dnl_val = std::get<1>(t);
        double inl_val = std::get<2>(t);

        double A1 = std::pow(dnl_val / dnl_ref, 2.0);
        double X  = std::log2(A1);
        double A2 = std::pow(inl_val / inl_ref, 2.0);
        double Y05= std::log2(A2);
        double A2_INL1 = A2 / 4.0;
        double Y10     = std::log2(A2_INL1);
        double A_dig = 0.0;
        int cnt_h = (1<<th)-1;
        int cnt_l = (1<<tl)-1;
        if(th>0) A_dig += cnt_h * S_unit;
        if(tl>0) A_dig += cnt_l * S_unit;

        double A_ana_max_INL1 = std::max(A1, A2_INL1);
        double A_chip_INL1    = A_ana_max_INL1 + A_dig;
        double log2_Achip_INL1     = std::log2(A_chip_INL1);

        table.push_back({m, th, tl, dnl_val, A1, X, inl_val, A2, Y05, A2_INL1, Y10, A_dig, A_chip_INL1, log2_Achip_INL1});
    }
    return table;
}

void savePaperCSV(const std::vector<ResRow>& table, const std::string& fname)
{
    std::ofstream ofs(fname);
    ofs<<std::fixed; ofs.precision(6);
    ofs<<"m(high therm bits),therm_high,therm_low,DNL_max_RMS[LSB],A1,X_log2A1,INL_max_RMS[LSB],A2,Y05_log2A2,A2_INL1,Y10_log2A2INL1,A_digital,A_chip_INL1,log2_Achip_INL1\n";
    for(auto &r:table)
    {
        ofs<<r.m<<","<<r.therm_high<<","<<r.therm_low<<","
           <<r.dnl_val<<","<<r.A1<<","<<r.X<<","
           <<r.inl_val<<","<<r.A2<<","<<r.Y05<<","
           <<r.A2_INL1<<","<<r.Y10<<","
           <<r.A_digital<<","<<r.A_chip_INL1<<","<<r.log2_Achip_INL1<<"\n";
    }
    ofs.close();
    std::cout<<"Paper table saved: "<<fname<<"\n";
}

void drawPaperPlots(const std::vector<ResRow>& table)
{
    TFile* fout = new TFile("paper_segDAC.root","RECREATE");

    TCanvas* c_inl = new TCanvas("c_inl","SegAll INL RMS",900,600);
    c_inl->SetFillColor(kWhite);
    c_inl->cd(); gPad->SetGrid();
    TGraph* g_inl_rms = new TGraph();
    for(auto &r : table)
    {
        if(r.m == 0) continue; 
        g_inl_rms->SetPoint(g_inl_rms->GetN(), r.m, r.inl_val);
    }
    g_inl_rms->SetLineColor(kBlack);
    g_inl_rms->SetLineWidth(2);
    g_inl_rms->SetMarkerStyle(8);
    g_inl_rms->SetMarkerSize(0.5);
    g_inl_rms->Draw("ALP");
    g_inl_rms->GetXaxis()->SetTitle("m (High thermometer bits)");
    g_inl_rms->GetYaxis()->SetTitle("INL_{RMS,max} [LSB]");
    g_inl_rms->GetXaxis()->SetLimits(1,9);
    g_inl_rms->GetYaxis()->SetRangeUser(0.0,0.8); 
    c_inl->SaveAs("SegAll_INL_rms.png");
    c_inl->Write();

    TCanvas* c_dnl_area = new TCanvas("c_dnl_area","DNL RMS max",900,600);
    c_dnl_area->SetFillColor(kWhite);
    c_dnl_area->cd();
    gPad->SetGrid();
    TGraph* g_dnl_rms = new TGraph();
    for(auto &r : table)
    {
        if(r.m ==0) continue;
        g_dnl_rms->SetPoint(g_dnl_rms->GetN(), r.m, r.dnl_val);
    }
    g_dnl_rms->SetLineColor(kBlack);
    g_dnl_rms->SetLineWidth(2);
    g_dnl_rms->SetMarkerStyle(8);
    g_dnl_rms->SetMarkerSize(0.5);
    g_dnl_rms->Draw("ALP");
    g_dnl_rms->GetXaxis()->SetTitle("m (High thermometer bits)");
    g_dnl_rms->GetYaxis()->SetTitle("DNL_{RMS,max} [LSB]");
    g_dnl_rms->GetXaxis()->SetLimits(1,9);
    TLegend* leg2 = new TLegend(0.15,0.72,0.38,0.88);
    leg2->AddEntry(g_dnl_rms,"DNL_{RMS,max}","lp");
    leg2->SetFillStyle(0);
    leg2->Draw();
    c_dnl_area->SaveAs("SegAll_DNL_only.png");
    c_dnl_area->Write();
    TCanvas* c_trade = new TCanvas("c_trade","Area trade‑off",960,650);
    c_trade->SetFillColor(kWhite);
    c_trade->cd(); gPad->SetGrid();
    TGraph* gX=new TGraph();
    TGraph* gY05=new TGraph();
    TGraph* gY10=new TGraph();
    TGraph* gDigLog=new TGraph();
    TGraph* gChipLog=new TGraph();
    for(auto &r : table)
    {
        if(r.m ==0) continue;
        gX->SetPoint(gX->GetN(),r.m, r.X);
        gY05->SetPoint(gY05->GetN(),r.m, r.Y05);
        gY10->SetPoint(gY10->GetN(),r.m, r.Y10);
        double ld = (r.A_digital>0) ? std::log2(r.A_digital) : 0.0;
        gDigLog->SetPoint(gDigLog->GetN(),r.m, ld);
        gChipLog->SetPoint(gChipLog->GetN(),r.m, r.log2_Achip_INL1);
    }
    gX->SetLineWidth(2); gX->SetLineColor(kBlack);
    gY05->SetLineWidth(2); gY05->SetLineColor(kBlue);
    gY10->SetLineWidth(2); gY10->SetLineColor(kMagenta); gY10->SetLineStyle(2);
    gDigLog->SetLineWidth(2); gDigLog->SetLineColor(kGreen); gDigLog->SetLineStyle(2);
    gChipLog->SetLineWidth(3); gChipLog->SetLineColor(kRed);

    gX->Draw("AL");
    gY05->Draw("L same");
    gY10->Draw("L same");
    gDigLog->Draw("L same");
    gChipLog->Draw("L same");

    TLegend* leg3 = new TLegend(0.12,0.15,0.42,0.35);
    leg3->AddEntry(gX,"X=log2(A1) DNL constraint","l");
    leg3->AddEntry(gY05,"Y05=log2(A2) INL 0.5LSB","l");
    leg3->AddEntry(gY10,"Y10=log2(A2/4) INL 1LSB","l");
    leg3->AddEntry(gDigLog,"log2(A_digital)","l");
    leg3->AddEntry(gChipLog,"log2(A_chip_INL1) total area","l");
    leg3->SetFillStyle(0);
    leg3->Draw();
    gX->GetXaxis()->SetTitle("m (High thermometer bits)");
    gX->GetYaxis()->SetTitle("log2(A/A_{unit})");
    gX->GetXaxis()->SetLimits(1,9);
    c_trade->SaveAs("SegAll_area_tradeoff.png");
    c_trade->Write();

    fout->Close();
    delete fout;
}

int main()
{
    std::default_random_engine rng(time(0));

    auto raw = simulateAllPaperDAC(rng);
    auto tab = computeAreaTablePaper(raw);
    savePaperCSV(tab, "paper_dac_table.csv");
    drawPaperPlots(tab);

    std::cout<<"Done! Output:\n";
    std::cout<<"  paper_dac_table.csv\n";
    std::cout<<"  fig3_4_INL_rms.png\n";
    std::cout<<"  fig3_5_DNL_digitalArea.png\n";
    std::cout<<"  paper_area_tradeoff.png\n";
    return 0;
}
