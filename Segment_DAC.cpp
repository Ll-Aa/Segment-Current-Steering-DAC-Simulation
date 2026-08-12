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
    int Mv;
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

std::tuple<double,double> simulateOneSegmentDAC(int M, std::default_random_engine& rng)
{
    int L = Nbit - M;
    int unit_per_therm_step = 1 << L;
    int num_therm_step      = (1 << M) - 1;
    int total_therm_unit    = num_therm_step * unit_per_therm_step;
    int num_low_bin_unit    = (1 << L) - 1;
    if( (total_therm_unit + num_low_bin_unit) != N_unit_total )
    {
        std::cerr<<"Warning: M="<<M<<" unit count mismatch!\n";
    }
    Eigen::MatrixXd dnl_seg_all(N_mc, N_dnl);
    Eigen::MatrixXd inl_seg_all(N_mc, N_code);
    std::normal_distribution<double> gauss(1.0, sigma_u);
    for(int mc = 0; mc < N_mc; mc++)
    {
        Eigen::VectorXd I_unit(N_unit_total);
        for(int iu = 0; iu < N_unit_total; iu++)
        {
            I_unit(iu) = gauss(rng);
        }
        int ptr = 0;
        Eigen::VectorXd I_therm_seg(num_therm_step);
        for(int i = 0; i < num_therm_step; i++)
        {
            double sum = 0.0;
            for(int s = 0; s < unit_per_therm_step; s++)
            {
                sum += I_unit(ptr++);
            }
            I_therm_seg(i) = sum;
        }
        Eigen::VectorXd I_bin_low(L);
        for(int k = 0; k < L; k++)
        {
            int w = 1 << (L - 1 - k);
            double sum = 0.0;
            for(int s = 0; s < w; s++)
            {
                sum += I_unit(ptr++);
            }
            I_bin_low(k) = sum;
        }
        Eigen::VectorXd Iout_seg(N_code);
        for(int code = 0; code < N_code; code++)
        {
            int high_val = code >> L;
            int low_val  = code & ((1 << L) - 1);
            double Itotal = 0.0;
            if(high_val > 0)
            {
                Itotal += I_therm_seg.head(high_val).sum();
            }
            for(int kk = 0; kk < L; kk++)
            {
                if( (low_val >> (L-1-kk)) & 1 )
                {
                    Itotal += I_bin_low(kk);
                }
            }
            Iout_seg(code) = Itotal;
        }
        Eigen::VectorXd dnl_seg = calDNL(Iout_seg);
        Eigen::VectorXd inl_seg = calINL_endpoint(Iout_seg);
        dnl_seg_all.row(mc) = dnl_seg;
        inl_seg_all.row(mc) = inl_seg;
    }
    Eigen::VectorXd dnl_code_rms = rmsAlongMC(dnl_seg_all);
    double dnl_rms_max = dnl_code_rms.maxCoeff();
    Eigen::VectorXd inl_code_rms = rmsAlongMC(inl_seg_all);
    double inl_rms_max = inl_code_rms.maxCoeff();
    return {dnl_rms_max, inl_rms_max};
}

std::vector<std::tuple<int,double,double>> simulateAllSegmentDAC(std::default_random_engine& rng)
{
    std::vector<std::tuple<int,double,double>> out;
    for(int M = 0; M <= 12; M++)
    {
        auto [d,i] = simulateOneSegmentDAC(M, rng);
        out.emplace_back(M, d, i);
    }
    return out;
}

std::vector<ResRow> computeAreaTable(const std::vector<std::tuple<int,double,double>>& raw)
{
    int idx_120 = 0;
    for(int i=0;i<(int)raw.size();i++)
    {
        if(std::get<0>(raw[i]) == 12){ idx_120=i; break; }
    }
    double dnl_ref  = std::get<1>(raw[idx_120]);
    double inl_ref  = std::get<1>(raw[idx_120]);
    std::vector<ResRow> table;
    for(auto &row : raw)
    {
        int Mv      = std::get<0>(row);
        double dnl_val = std::get<1>(row);
        double inl_val = std::get<2>(row);
        double A1 = std::pow( dnl_val / dnl_ref, 2.0 );
        double X  = std::log2(A1);
        double A2 = std::pow( inl_val / inl_ref, 2.0 );
        double Y05= std::log2(A2);
        double A2_INL1 = A2 / 4.0;
        double Y10     = std::log2(A2_INL1);
        double A_digital = 0.0;
        if(Mv !=0)
        {
            double num_unit = (1 << Mv) - 1;
            A_digital = num_unit * S_unit;
        }
        double A_ana_max_INL1 = std::max(A1, A2_INL1);
        double A_chip_INL1    = A_ana_max_INL1 + A_digital;
        double log2_Achip_INL1= std::log2(A_chip_INL1);
        table.push_back({Mv,dnl_val,A1,X,inl_val,A2,Y05,A2_INL1,Y10,A_digital,A_chip_INL1,log2_Achip_INL1});
    }
    return table;
}

std::tuple<int,int,double> findOptimalM(const std::vector<ResRow>& table)
{
    std::vector<double> log2_Achip_vec;
    std::vector<int> M_data_vec;
    for(auto &r : table)
    {
        log2_Achip_vec.push_back(r.log2_Achip_INL1);
        M_data_vec.push_back(r.Mv);
    }

    int idx_min_global = 0;
    double val_min = table[0].A_chip_INL1;
    for(int i=1;i<(int)table.size();i++)
    {
        if(table[i].A_chip_INL1 < val_min)
        {
            val_min = table[i].A_chip_INL1;
            idx_min_global = i;
        }
    }

    int plat_start_idx = -1;
    for(int i=1; i<(int)log2_Achip_vec.size();i++)
    {
        double delta = log2_Achip_vec[i] - log2_Achip_vec[i-1];
        int cand_idx = i-1;
        int cand_M = M_data_vec[cand_idx];
        if(delta >= delta_thresh && cand_M > 6)
        {
            plat_start_idx = cand_idx;
            break;
        }
    }

    if(plat_start_idx >=0)
    {
        return { M_data_vec[plat_start_idx], plat_start_idx, log2_Achip_vec[plat_start_idx] };
    }
    else
    {
        return { M_data_vec[idx_min_global], idx_min_global, log2_Achip_vec[idx_min_global] };
    }
}
void saveAreaTableToCSV(const std::vector<ResRow>& table,
                        const std::string& filename,
                        int M_opt)
{
    std::ofstream ofs(filename);
    ofs << std::fixed;
    ofs.precision(6);
    ofs << "M (Thermometer bits),"
        << "L (Binary bits),"
        << "DNL_max_RMS [LSB],"
        << "A1 (DNL area ratio),"
        << "X = log2(A1),"
        << "INL_max_RMS [LSB],"
        << "A2 (INL area ratio 0.5LSB),"
        << "Y05 = log2(A2),"
        << "A2_INL1 (INL 1LSB),"
        << "Y10 = log2(A2_INL1),"
        << "A_digital,"
        << "A_chip (INL<=1LSB),"
        << "log2(A_chip)\n";
    for(auto &r : table)
    {
        int Lv = Nbit - r.Mv;
        ofs << r.Mv << ","
            << Lv << ","
            << r.dnl_val << ","
            << r.A1 << ","
            << r.X << ","
            << r.inl_val << ","
            << r.A2 << ","
            << r.Y05 << ","
            << r.A2_INL1 << ","
            << r.Y10 << ","
            << r.A_digital << ","
            << r.A_chip_INL1 << ","
            << r.log2_Achip_INL1 << "\n";
    }
    ofs << "\n";
    ofs << "Optimal M (engineering),," << M_opt << "\n";
    ofs << "Delta threshold,," << delta_thresh << "\n";
    ofs.close();
    std::cout << "Area table saved to: " << filename << "\n";
}

void drawAreaTradeoff(const std::vector<ResRow>& table, int M_opt, double log2A_opt)
{
    TFile* fout = new TFile("segDAC_sim.root","RECREATE");
    TCanvas* c3 = new TCanvas("c3","Area trade-off",980,700);
    c3->SetFillColor(kWhite);
    c3->cd(); gPad->SetGrid();

    int nPt = (int)table.size();
    TGraph* g1 = new TGraph(nPt);
    TGraph* g2 = new TGraph(nPt);
    TGraph* g3 = new TGraph(nPt);
    TGraph* g4 = new TGraph(nPt);
    TGraph* g5 = new TGraph(nPt);

    for(int i=0;i<nPt;i++){
      auto &r = table[i];
      g1->SetPoint(i, double(r.Mv), r.X);
      g2->SetPoint(i, double(r.Mv), r.Y05);
      g3->SetPoint(i, double(r.Mv), r.Y10);
      double ladig = (std::log2(r.A_digital) > 0) ? std::log2(r.A_digital) : 0;
      g4->SetPoint(i, double(r.Mv), ladig);
      g5->SetPoint(i, double(r.Mv), r.log2_Achip_INL1);
    }
    g1->SetLineWidth(1.5);
    g1->SetMarkerStyle(0);       
    g2->SetLineWidth(1.5);
    g2->SetMarkerStyle(0);

    g3->SetLineWidth(1.6);
    g3->SetLineColor(kMagenta);
    g3->SetLineStyle(2);     

    g4->SetLineWidth(3);
    g4->SetLineColor(kGreen);
    g4->SetLineStyle(2);     

    g5->SetLineWidth(2.2);  
    g5->SetLineColor(kBlue);

    g1->Draw("AL");
    g2->Draw("L");
    g3->Draw("L");
    g4->Draw("L");
    g5->Draw("L");

    TGraph* g_opt = new TGraph(1);
    g_opt->SetPoint(0, double(M_opt), log2A_opt);
    g_opt->SetMarkerColor(kRed);
    g_opt->SetMarkerStyle(20);
    g_opt->SetMarkerSize(2.2);   
    g_opt->Draw("P");

    TLegend *leg = new TLegend(0.12,0.25,0.38,0.4);
    leg->AddEntry(g1,"X=log_{2}(A_{1}) DNL constraint(0.5LSB)","l");
    leg->AddEntry(g2,"Y=log_{2}(A_{2}) INL=0.5LSB","l");
    leg->AddEntry(g3,"Y=log_{2}(A_{2}/4) INL=1.0LSB","l");
    leg->AddEntry(g4,"log_{2}(A_{digital}) digital area","l");
    leg->AddEntry(g5,"log_{2}(A_{chip}) total chip area INL<=1.0LSB","l");
    leg->SetFillStyle(1);     
    leg->SetBorderSize(1);
    leg->Draw();

    g1->GetXaxis()->SetTitle("M (Thermometer bits)");
    g1->GetYaxis()->SetTitle("log_{2}(A/A_{unit})");
    g1->GetXaxis()->SetTitleSize(0.045);
    g1->GetYaxis()->SetTitleSize(0.045);
    g1->GetXaxis()->SetLimits(0,12);
    c3->SaveAs("segDAC_area_tradeoff.png");
    c3->Write();
    fout->Close();
    delete fout;
}

int main()
{
    std::default_random_engine rng(time(0));
    auto raw_table = simulateAllSegmentDAC(rng);
    auto res_table = computeAreaTable(raw_table);
    auto [M_opt, idx_opt, log2A_opt] = findOptimalM(res_table);
    saveAreaTableToCSV(res_table, "segDAC_area_table.csv", M_opt);
    std::cout << "Engineering optimal thermometer bits M = " << M_opt << "\n";
    drawAreaTradeoff(res_table, M_opt, log2A_opt);

    return 0;
}
